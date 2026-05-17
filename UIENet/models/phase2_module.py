"""
Phase 2: 多域特征协同提取 v4.0

四个分支各接收 Phase1 的分支专用投影，实现真正的多域特异性：
- A (Swin): dilation=2 大感受野 → 全局语义
- B (Freq): Identity → FFT 频域解耦
- C (CBAM): 3x3 局部 + FB 频域引导 → 空域注意力
- D (UNet): 1x1 保真 → 多尺度细节修复
"""

import torch, torch.nn as nn
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

    def forward(self, phase1_out):
        fa = self.branch_a(phase1_out["fa"])
        fb, pred_b_mid = self.branch_b(phase1_out["fb"])
        fc = self.branch_c(phase1_out["fc"], fb)
        fd, pred_d_mid = self.branch_d(phase1_out["fd"])
        return {"fa":fa,"fb":fb,"fc":fc,"fd":fd,
                "pred_b_mid":pred_b_mid,"pred_d_mid":pred_d_mid}
