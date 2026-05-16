"""
Phase 1: 物理感知与预处理

包含两个子模块：
1. Retinex 分解模块 —— 轻量 CNN，将输入分解为光照层 (Illumination) 和反射层 (Reflectance)
2. 密度图估计器 —— 轻量 CNN，估计水体浑浊度/散射密度地图

两个模块并行处理输入图像，输出拼接后经 1×1 卷积压缩，形成物理先验特征，
送入 Phase 2 的四个分支。

设计动机：
- Retinex 分解显式建模水下图像的照度-反射分离，为后续增强提供物理约束
- 浑浊度地图为网络提供各区域的退化程度先验，指导后续模块的关注重点
- 所有超参数从 config.yaml 通过点号访问，如 config.phase1_feat_channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinexDecompose(nn.Module):
    """
    Retinex 分解模块 v3.0
    将输入图像分解为 Illumination (光照层) 和 Reflectance (反射层)
    加深网络 + 残差连接，提升分解精度。
    """
    def __init__(self, in_channels=3, mid_channels=48, out_channels=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # 两路并行输出头
        self.illum_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.reflect_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.shared(x)
        illumination = self.illum_head(feat)
        reflectance = self.reflect_head(feat)
        return illumination, reflectance


class TurbidityEstimator(nn.Module):
    """
    浑浊度/密度图估计器 v3.0
    加深网络 + 残差块，提升密度估计精度。
    """
    def __init__(self, in_channels=3, mid_channels=96):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        # 残差块 × 2
        self.res1 = self._make_res_block(mid_channels)
        self.res2 = self._make_res_block(mid_channels)
        self.head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def _make_res_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        feat = self.stem(x)
        feat = F.relu(feat + self.res1(feat), inplace=True)
        feat = F.relu(feat + self.res2(feat), inplace=True)
        return self.head(feat)


class Phase1(nn.Module):
    """
    Phase 1 总体模块
    整合 Retinex 分解和浑浊度估计，生成物理先验特征

    输入:  RGB 图像 (B, 3, H, W)
    输出:  字典 {
        'prior_feat':    物理先验特征 (B, C, H, W), C = feat_channels (默认64)
        'illumination':  光照层 (B, 3, H, W)
        'reflectance':   反射层 (B, 3, H, W)
        'turbidity':     浑浊度地图 (B, 1, H, W)
    }
    """
    def __init__(self, config):
        super().__init__()
        # 【修改】使用点号访问 config，不再使用 .get()
        in_channels = config.in_channels
        retinex_mid = config.phase1_retinex_mid_channels
        density_mid = config.phase1_density_mid_channels
        feat_channels = config.phase1_feat_channels

        self.retinex = RetinexDecompose(in_channels, retinex_mid, in_channels)
        self.turbidity = TurbidityEstimator(in_channels, density_mid)

        # 【修改】拼接 Illum(3) + Reflect(3) + Turbidity(1) = 7 通道 → 1x1 Conv + ReLU 压缩到 C
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 1, feat_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        illumination, reflectance = self.retinex(x)
        turbidity = self.turbidity(x)

        # 物理先验特征：拼接后 1×1 压缩
        prior_feat = self.compress(torch.cat([illumination, reflectance, turbidity], dim=1))

        return {
            'prior_feat': prior_feat,
            'illumination': illumination,
            'reflectance': reflectance,
            'turbidity': turbidity
        }