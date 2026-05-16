"""
分支 D: 空域-感知分支（轻量 U-Net）

采用轻量化的 U-Net 结构，通过多尺度编解码和跨层跳连捕获细节信息，
弥补其他分支可能丢失的空域高频细节。训练时额外输出中间监督图 Pred_D_mid，
通过 self.training 控制，推理时仅返回主特征和 None。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BranchUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        base = config.branch_unet.base_channels
        self.input_size = config.branch_unet.input_size

        # 编码器（4 级下采样）
        self.enc1 = ConvBlock(base, base)                      # H, W
        self.enc2 = ConvBlock(base, base * 2, stride=2)        # H/2, W/2
        self.enc3 = ConvBlock(base * 2, base * 4, stride=2)    # H/4, W/4
        self.enc4 = ConvBlock(base * 4, base * 8, stride=2)    # H/8, W/8  [新增]

        # 瓶颈
        self.bottleneck = nn.Sequential(
            ConvBlock(base * 8, base * 8),
            ConvBlock(base * 8, base * 8),
        )

        # 解码器（逐步上采样 + 跳连）
        self.up4 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            ConvBlock(base * 8, base * 4),   # 跳连拼接后 8C→4C
            ConvBlock(base * 4, base * 4)
        )
        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBlock(base * 4, base * 2),   # 跳连拼接后 4C→2C
            ConvBlock(base * 2, base * 2)
        )
        self.up2 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBlock(base * 2, base),        # 跳连拼接后 2C→C
            ConvBlock(base, base)
        )

        # 中间监督头（从 dec3 中间层输出）
        self.mid_supervision_head = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        feat1 = self.enc1(x)          # (B, C,   H,   W)
        feat2 = self.enc2(feat1)      # (B, 2C,  H/2, W/2)
        feat3 = self.enc3(feat2)      # (B, 4C,  H/4, W/4)
        feat4 = self.enc4(feat3)      # (B, 8C,  H/8, W/8)

        bn = self.bottleneck(feat4)   # (B, 8C,  H/8, W/8)

        up4 = self.up4(bn)            # (B, 4C,  H/4, W/4)
        dec4_input = torch.cat([up4, feat3], dim=1)  # (B, 8C, H/4, W/4)
        dec4_out = self.dec4(dec4_input)             # (B, 4C, H/4, W/4)

        up3 = self.up3(dec4_out)      # (B, 2C,  H/2, W/2)
        dec3_input = torch.cat([up3, feat2], dim=1)  # (B, 4C, H/2, W/2)
        mid_feat = self.dec3(dec3_input)             # (B, 2C, H/2, W/2)

        up2 = self.up2(mid_feat)      # (B, C,   H,   W)
        dec2_input = torch.cat([up2, feat1], dim=1)  # (B, 2C, H, W)
        fd = self.dec2(dec2_input)    # (B, C,   H,   W)

        if self.training:
            pred_d_mid = self.mid_supervision_head(mid_feat)
            pred_d_mid = F.interpolate(pred_d_mid, size=tuple(self.input_size),
                                       mode='bilinear', align_corners=False)
            return fd, pred_d_mid
        else:
            return fd, None