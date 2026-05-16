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
        base = config.branch_unet.base_channels  # 64
        self.input_size = config.branch_unet.input_size

        # 编码器（逐级下采样）
        self.enc1 = ConvBlock(base, base)              # 不改变尺寸
        self.enc2 = ConvBlock(base, base * 2, stride=2)  # 1/2
        self.enc3 = ConvBlock(base * 2, base * 4, stride=2)  # 1/4

        # 瓶颈
        self.bottleneck = ConvBlock(base * 4, base * 4)

        # 解码器（上采样 + 跳连）
        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBlock(base * 4, base * 2),  # 输入为跳连拼接后的 4C
            ConvBlock(base * 2, base * 2)
        )
        self.up2 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBlock(base * 2, base),       # 跳连拼接后 2C
            ConvBlock(base, base)
        )

        # 中间监督头（仅训练时使用）
        self.mid_supervision_head = nn.Sequential(
            nn.Conv2d(base * 2, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        # 编码器
        feat1 = self.enc1(x)          # (B, C, H, W)
        feat2 = self.enc2(feat1)      # (B, 2C, H/2, W/2)
        feat3 = self.enc3(feat2)      # (B, 4C, H/4, W/4)

        # 瓶颈
        bn = self.bottleneck(feat3)   # (B, 4C, H/4, W/4)

        # 解码器，逐步上采样并跳连
        up3 = self.up3(bn)            # (B, 2C, H/2, W/2)
        dec3_input = torch.cat([up3, feat2], dim=1)  # (B, 4C, H/2, W/2)
        mid_feat = self.dec3(dec3_input)   # (B, 2C, H/2, W/2) 中间层特征

        up2 = self.up2(mid_feat)      # (B, C, H, W)
        dec2_input = torch.cat([up2, feat1], dim=1)  # (B, 2C, H, W)
        fd = self.dec2(dec2_input)    # (B, C, H, W)  最终输出

        if self.training:
            # 中间监督：从 mid_feat 生成 3 通道图并上采样到原图尺寸
            pred_d_mid = self.mid_supervision_head(mid_feat)
            pred_d_mid = F.interpolate(pred_d_mid, size=self.input_size,
                                       mode='bilinear', align_corners=False)
            return fd, pred_d_mid
        else:
            return fd, None