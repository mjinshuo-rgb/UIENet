"""
分支 A: 图域推理分支（Swin Transformer）

自实现 Swin Transformer Block，不依赖 timm 的内部接口，
支持任意输入分辨率，并通过 shift_size 控制 W-MSA 和 SW-MSA 交替。

优化说明：
- _get_attn_mask 在首次计算后缓存于 self._attn_mask_cache，
  避免每次 forward 重复计算相同尺寸的 mask。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):
    """窗口多头注意力（支持相对位置偏置）"""
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        ws = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads)
        )
        coords = torch.stack(torch.meshgrid([torch.arange(ws), torch.arange(ws)]))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += ws - 1
        relative_coords[:, :, 1] += ws - 1
        relative_coords[:, :, 0] *= 2 * ws - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class SwinBlock(nn.Module):
    """单个 Swin Transformer Block：窗口注意力 + MLP，含 mask 缓存"""
    def __init__(self, dim, num_heads, window_size, shift_size=0, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )
        # 注意力掩码缓存字典，key为 (Hp, Wp)
        self._attn_mask_cache = {}

    def forward(self, x, H, W):
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        shifted_x = F.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = shifted_x.shape[1], shifted_x.shape[2]

        x_windows = shifted_x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)

        attn_mask = self._get_attn_mask(Hp, Wp) if self.shift_size > 0 else None
        attn_out = self.attn(x_windows, attn_mask)

        attn_out = attn_out.view(B, Hp // ws, Wp // ws, ws, ws, C)
        attn_out = attn_out.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
        attn_out = attn_out[:, :H, :W, :].contiguous()

        if self.shift_size > 0:
            attn_out = torch.roll(attn_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        attn_out = attn_out.view(B, H * W, C)
        x = shortcut + self.drop_path(attn_out)

        shortcut = x
        x = shortcut + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def _get_attn_mask(self, Hp, Wp):
        key = (Hp, Wp)
        if key not in self._attn_mask_cache:
            device = next(self.parameters()).device
            ws = self.window_size
            img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
            h_slices = (slice(0, -ws), slice(-ws, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -ws), slice(-ws, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = img_mask.view(1, Hp // ws, ws, Wp // ws, ws, 1)
            mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self._attn_mask_cache[key] = attn_mask
        return self._attn_mask_cache[key]


class BranchSwin(nn.Module):
    """Swin Transformer 分支，多个 SwinBlock 堆叠"""
    def __init__(self, config):
        super().__init__()
        dim = config.branch_swin.embed_dim
        window_size = config.branch_swin.window_size
        num_heads = config.branch_swin.num_heads
        num_layers = config.branch_swin.num_layers
        mlp_ratio = config.branch_swin.mlp_ratio
        drop_rate = config.branch_swin.drop_rate
        attn_drop_rate = config.branch_swin.attn_drop_rate

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            block = SwinBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            self.blocks.append(block)

        self.input_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.input_proj(x)
        x = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            x = block(x, H, W)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x