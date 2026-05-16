"""
Phase 3: 智能融合与重构

包含两个子模块：
1. Patch-based Cross-Attention 特征融合
   以 FA 为 Query，分别与 FB/FC/FD 做局部 Cross-Attention，
   patch_size=8，支持任意尺寸输入（自动 padding）。
2. LAB 色彩空间自适应校正
   将融合特征映射为 LAB 空间的 L 和 ab 残差，通过可微分
   的 RGB↔LAB 转换实现端到端色彩校正。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import kornia


class PatchCrossAttention(nn.Module):
    """Patch-based Cross-Attention，支持任意输入尺寸（自动 padding）"""
    def __init__(self, dim, num_heads=4, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query_feat, kv_feat):
        B, C, H, W = query_feat.shape
        P = self.patch_size

        # 确保能被 patch_size 整除
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        if pad_h > 0 or pad_w > 0:
            query_feat = F.pad(query_feat, (0, pad_w, 0, pad_h))
            kv_feat = F.pad(kv_feat, (0, pad_w, 0, pad_h))
        Hp, Wp = query_feat.shape[2], query_feat.shape[3]

        q = einops.rearrange(query_feat, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c',
                             p1=P, p2=P)
        k = einops.rearrange(kv_feat,   'b c (h p1) (w p2) -> b (h w) (p1 p2) c',
                             p1=P, p2=P)
        v = k

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = einops.rearrange(q, 'b n t (h d) -> b n h t d', h=self.num_heads)
        k = einops.rearrange(k, 'b n t (h d) -> b n h t d', h=self.num_heads)
        v = einops.rearrange(v, 'b n t (h d) -> b n h t d', h=self.num_heads)

        attn = torch.einsum('bnhtd,bnhsd->bnhts', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bnhts,bnhsd->bnhtd', attn, v)

        out = einops.rearrange(out, 'b n h t d -> b n t (h d)')
        out = self.out_proj(out)
        out = einops.rearrange(out, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
                               h=Hp//P, w=Wp//P, p1=P, p2=P)

        # 裁剪回原尺寸
        out = out[:, :, :H, :W].contiguous()
        return out


class FusionModule(nn.Module):
    """特征融合模块：以 FA 为 Query，分别与 FB、FC、FD 做 Patch Cross-Attention"""
    def __init__(self, dim=64, num_heads=4, patch_size=8):
        super().__init__()
        self.ca_ab = PatchCrossAttention(dim, num_heads, patch_size)
        self.ca_ac = PatchCrossAttention(dim, num_heads, patch_size)
        self.ca_ad = PatchCrossAttention(dim, num_heads, patch_size)
        self.fuse_conv = nn.Conv2d(dim * 4, dim, 1)

    def forward(self, fa, fb, fc, fd):
        out_ab = self.ca_ab(fa, fb)
        out_ac = self.ca_ac(fa, fc)
        out_ad = self.ca_ad(fa, fd)
        fused = torch.cat([fa, out_ab, out_ac, out_ad], dim=1)
        return self.fuse_conv(fused)


class LABColorCorrection(nn.Module):
    """LAB 色彩空间自适应校正，使用可微分 RGB↔LAB 转换"""
    def __init__(self, feat_channels=64):
        super().__init__()
        self.l_residual = nn.Sequential(
            nn.Conv2d(feat_channels, 1, 3, padding=1),
            nn.Tanh()
        )
        self.ab_residual = nn.Sequential(
            nn.Conv2d(feat_channels, 2, 3, padding=1),
            nn.Tanh()
        )
        # 残差缩放因子（初始值调大以保证有效色彩校正）
        self.l_scale = nn.Parameter(torch.tensor(5.0))
        self.ab_scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, fused_feat, input_rgb):
        input_rgb_01 = (input_rgb + 1.0) / 2.0
        lab = kornia.color.rgb_to_lab(input_rgb_01)

        L = lab[:, 0:1, :, :]
        ab = lab[:, 1:3, :, :]

        delta_L = self.l_residual(fused_feat) * self.l_scale
        delta_ab = self.ab_residual(fused_feat) * self.ab_scale

        L_corrected = L + delta_L
        ab_corrected = ab + delta_ab

        lab_corrected = torch.cat([L_corrected, ab_corrected], dim=1)
        rgb_corrected = kornia.color.lab_to_rgb(lab_corrected).clamp(0, 1)
        rgb_corrected = rgb_corrected * 2.0 - 1.0
        return rgb_corrected


class Phase3(nn.Module):
    """Phase 3 总体模块：特征融合 + LAB 色彩校正"""
    def __init__(self, config):
        super().__init__()
        dim = config.phase3.dim
        num_heads = config.phase3.num_heads
        patch_size = config.phase3.patch_size

        self.fusion = FusionModule(dim, num_heads, patch_size)
        self.color_correction = LABColorCorrection(dim)

    def forward(self, fa, fb, fc, fd, input_rgb):
        fused_feat = self.fusion(fa, fb, fc, fd)
        output_rgb = self.color_correction(fused_feat, input_rgb)
        return output_rgb