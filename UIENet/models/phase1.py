"""
Phase 1: 可学习物理先验提取 v4.1

三个可学习物理先验：
1. Retinex 分解（I x R approx input，自监督重建约束）
2. 暗通道 + 密度估计器（硬先验辅助，提升浑浊度估计精度）
3. 3 级 CNN 深度特征提取 + 4 分支专用投影

对比 v4.0：恢复 Retinex 作为可学习先验，暗通道仅辅助浑浊度估计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinexDecompose(nn.Module):
    """Retinex 分解: 输入 [-1,1] -> illumination [0,1], reflectance [0,1]"""
    def __init__(self, in_channels=3, mid_channels=48, out_channels=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.illum_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, out_channels, 1),
            nn.Sigmoid()
        )
        self.reflect_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.shared(x)
        return self.illum_head(feat), self.reflect_head(feat)


class DarkChannel(nn.Module):
    """暗通道: 15x15 min(R,G,B)，无参数"""
    def __init__(self, patch_size=15):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        x_01 = (x + 1.0) / 2.0
        P = self.patch_size; pad = P // 2
        x_pad = F.pad(x_01, (pad, pad, pad, pad), mode='reflect')
        patches = x_pad.unfold(2, P, 1).unfold(3, P, 1)
        dark = patches.min(dim=1)[0].min(dim=-1)[0].min(dim=-1)[0]
        return dark.unsqueeze(1)


class TurbidityEstimator(nn.Module):
    """浑浊度估计: RGB[0,1] + dark -> 密度图 [0,1]"""
    def __init__(self, in_channels=4, mid_channels=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
        )
        self.head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1),
            nn.BatchNorm2d(mid_channels // 2), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        f = self.stem(x)
        f = F.relu(f + self.res1(f), inplace=True)
        f = F.relu(f + self.res2(f), inplace=True)
        return self.head(f)


class DeepFeatureExtractor(nn.Module):
    """3级 CNN: 8ch(RGB+illum+reflect+dark+turbidity) -> 128ch -> 分支专用投影"""
    def __init__(self, in_channels=11, base=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.proj_swin = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
        )
        self.proj_freq = nn.Identity()
        self.proj_cbam = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
        )
        self.proj_unet = nn.Sequential(
            nn.Conv2d(base, base, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 1),
        )

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        feat = f1 + self.upsample(f3)
        return {
            'fa': self.proj_swin(feat), 'fb': self.proj_freq(feat),
            'fc': self.proj_cbam(feat), 'fd': self.proj_unet(feat),
        }


class Phase1(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_channels = config.phase1_feat_channels
        self.retinex = RetinexDecompose(3, config.phase1_retinex_mid_channels, 3)
        self.dark_channel = DarkChannel(patch_size=15)
        self.turbidity = TurbidityEstimator(in_channels=4, mid_channels=config.phase1_density_mid_channels)
        self.extractor = DeepFeatureExtractor(in_channels=11, base=feat_channels)

    def forward(self, x):
        x_01 = (x + 1.0) / 2.0
        illumination, reflectance = self.retinex(x_01)
        dark = self.dark_channel(x)
        turbidity = self.turbidity(torch.cat([x_01, dark], dim=1))
        prior_input = torch.cat([x_01, illumination, reflectance, dark, turbidity], dim=1)
        branch_feats = self.extractor(prior_input)
        return {
            'fa': branch_feats['fa'], 'fb': branch_feats['fb'],
            'fc': branch_feats['fc'], 'fd': branch_feats['fd'],
            'illumination': illumination, 'reflectance': reflectance,
            'turbidity': turbidity,
        }
