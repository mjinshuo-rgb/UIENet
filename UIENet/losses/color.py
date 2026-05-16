"""
LAB 色彩损失
在 LAB 空间仅对 ab 色度通道计算 L1 损失，
直接监督色彩校正效果，不干扰亮度优化。
"""

import torch
import torch.nn as nn
import kornia


class LABColorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_01 = (pred + 1.0) / 2.0
        target_01 = (target + 1.0) / 2.0
        pred_lab = kornia.color.rgb_to_lab(pred_01)
        target_lab = kornia.color.rgb_to_lab(target_01)
        # ab 通道归一化到 [-1,1] 附近再计算 L1
        pred_ab = pred_lab[:, 1:, :, :] / 110.0
        target_ab = target_lab[:, 1:, :, :] / 110.0
        return torch.nn.functional.l1_loss(pred_ab, target_ab)