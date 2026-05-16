"""
分支 C: 空域-注意力分支（CBAM）

接收物理先验特征和分支 B 的频域引导信号 FB，
通过 CBAM 双重注意力（通道 + 空间）聚焦关键区域。
频域分支提供的幅相信息天然对应注意力分布（色偏区域应重点关注），
因此将 FB 作为引导信号提升注意力生成的准确性。
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(attn)


class BranchCBAM(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.branch_swin.embed_dim
        reduction = config.branch_cbam.reduction
        spatial_kernel = config.branch_cbam.spatial_kernel

        # 将拼接后的特征（物理先验 + FB）压缩回 dim
        self.compress = nn.Conv2d(dim * 2, dim, kernel_size=1)

        self.channel_attn = ChannelAttention(dim, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

        # FFN 精炼 + 残差（双层增强）
        self.out_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, prior_feat, fb):
        # prior_feat: (B, C, H, W) 物理先验特征
        # fb:         (B, C, H, W) 分支 B 的输出引导
        combined = torch.cat([prior_feat, fb], dim=1)
        x = self.compress(combined)

        # CBAM 注意力
        channel_att = self.channel_attn(x)
        x = x * channel_att
        spatial_att = self.spatial_attn(x)
        x = x * spatial_att

        fc = self.out_conv(x) + x  # 残差连接
        return fc