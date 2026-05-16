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
        # 输入范围 [-1, 1] → [0, 1]
        pred = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0
        pred_lab = kornia.color.rgb_to_lab(pred)
        target_lab = kornia.color.rgb_to_lab(target)
        # 仅对 ab 通道（索引 1 和 2）计算损失
        return torch.nn.functional.l1_loss(pred_lab[:, 1:, :, :], target_lab[:, 1:, :, :])