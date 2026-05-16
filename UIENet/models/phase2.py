"""
Phase 2: 多域特征协同提取

整合四个分支：
- 分支 A (Swin):  图域推理，全局语义相关性
- 分支 B (Freq):  频域解耦，幅相分别处理
- 分支 C (CBAM):  空域注意力，接收分支B的频域引导
- 分支 D (UNet):  空域感知，多尺度细节修复

输入: 物理先验特征字典
输出: 字典包含 FA, FB, FC, FD, pred_b_mid, pred_d_mid
"""

import torch
import torch.nn as nn
from models.phase2.branch_swin import BranchSwin
from models.phase2.branch_freq import BranchFreq
from models.phase2.branch_cbam import BranchCBAM
from models.phase2.branch_unet import BranchUNet


class Phase2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.branch_a = BranchSwin(config)
        self.branch_b = BranchFreq(config)
        self.branch_c = BranchCBAM(config)
        self.branch_d = BranchUNet(config)

    def forward(self, prior_feat):
        # prior_feat: (B, C, H, W)

        # 分支 A
        fa = self.branch_a(prior_feat)

        # 分支 B（训练时返回 fb, pred_b_mid；推理时返回 fb, None）
        fb, pred_b_mid = self.branch_b(prior_feat)

        # 分支 C（接收物理先验特征和 FB 频域引导）
        fc = self.branch_c(prior_feat, fb)

        # 分支 D（训练时返回 fd, pred_d_mid；推理时返回 fd, None）
        fd, pred_d_mid = self.branch_d(prior_feat)

        return {
            'fa': fa,
            'fb': fb,
            'fc': fc,
            'fd': fd,
            'pred_b_mid': pred_b_mid,
            'pred_d_mid': pred_d_mid
        }