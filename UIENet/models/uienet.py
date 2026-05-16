"""
UIENet: 水下图像增强主网络

整合三阶段架构：
- Phase 1: 物理感知与预处理（Retinex + 浑浊度估计）
- Phase 2: 多域特征协同提取（Swin + FFT + CBAM + UNet）
- Phase 3: 智能融合与重构（Patch Cross-Attention + LAB 校正）

返回增强图像及必要的中间结果供损失计算。
"""

import torch
import torch.nn as nn
from models.phase1 import Phase1
from models.phase2 import Phase2
from models.phase3 import Phase3


class UIENet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.phase1 = Phase1(config)
        self.phase2 = Phase2(config)
        self.phase3 = Phase3(config)

    def forward(self, x):
        """
        x: 输入 RGB 图像 (B, 3, H, W)，范围 [-1, 1]

        返回字典:
        - 'output':          增强后图像 (B, 3, H, W)
        - 'illumination':    Retinex 光照层
        - 'reflectance':     Retinex 反射层
        - 'turbidity':       浑浊度地图
        - 'pred_b_mid':      分支 B 中间监督图（训练时有，推理时为 None）
        - 'pred_d_mid':      分支 D 中间监督图（训练时有，推理时为 None）
        """
        # Phase 1
        phase1_out = self.phase1(x)
        prior_feat = phase1_out['prior_feat']

        # Phase 2
        phase2_out = self.phase2(prior_feat)
        fa = phase2_out['fa']
        fb = phase2_out['fb']
        fc = phase2_out['fc']
        fd = phase2_out['fd']
        pred_b_mid = phase2_out['pred_b_mid']
        pred_d_mid = phase2_out['pred_d_mid']

        # Phase 3
        output = self.phase3(fa, fb, fc, fd, x)

        return {
            'output': output,
            'illumination': phase1_out['illumination'],
            'reflectance': phase1_out['reflectance'],
            'turbidity': phase1_out['turbidity'],
            'pred_b_mid': pred_b_mid,
            'pred_d_mid': pred_d_mid
        }