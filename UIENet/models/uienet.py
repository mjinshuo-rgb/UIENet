"""
UIENet: 水下图像增强主网络 v4.0

- Phase 1: 暗通道 + 浑浊度 + 多尺度CNN → 4分支专用投影
- Phase 2: Swin/FFT/CBAM/UNet 各接收专用特征
- Phase 3: 1×Cross-Attn + Conv融合 + LAB校正
"""

import torch
import torch.nn as nn
from models.phase1 import Phase1
from models.phase2_module import Phase2
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
        - 'turbidity':       浑浊度地图
        - 'pred_b_mid':      分支 B 中间监督图（训练时有，推理时为 None）
        - 'pred_d_mid':      分支 D 中间监督图（训练时有，推理时为 None）
        """
        # Phase 1: 深度物理先验提取
        phase1_out = self.phase1(x)

        # Phase 2: 多域特征协同（各分支接收专用投影）
        phase2_out = self.phase2(phase1_out)
        fa = phase2_out['fa']
        fb = phase2_out['fb']
        fc = phase2_out['fc']
        fd = phase2_out['fd']
        pred_b_mid = phase2_out['pred_b_mid']
        pred_d_mid = phase2_out['pred_d_mid']

        # Phase 3: 融合与重构
        output = self.phase3(fa, fb, fc, fd, x)

        return {
            'output': output,
            'turbidity': phase1_out['turbidity'],
            'pred_b_mid': pred_b_mid,
            'pred_d_mid': pred_d_mid
        }