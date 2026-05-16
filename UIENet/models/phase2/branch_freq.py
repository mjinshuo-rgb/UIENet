"""
分支 B: 频域解耦分支

通过 FFT 将特征分解为幅度谱和相位谱，分别处理：
- 幅度谱修正：对应水下色偏纠正（低频能量重分配）
- 相位谱修复：对应散射模糊修复（高频相位纠正）

训练时额外输出中间监督图 Pred_B_mid（上采样到原图尺寸），
推理时仅返回主特征和 None，通过 self.training 控制。

修改说明：FFT 后幅度谱数值范围较大，直接送入卷积会导致训练不稳定，
因此对幅度谱取 log 压缩后再处理，改善数值稳定性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BranchFreq(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mid_channels = config.branch_freq.mid_channels
        self.input_size = config.branch_freq.input_size  # [H, W]

        # 幅度处理子网络（残差块 + 1×1 卷积）
        self.amp_net = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=1),
        )
        # 相位处理子网络（残差块 + 1×1 卷积）
        self.phase_net = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=1),
        )
        # 输出卷积
        self.output_conv = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, 3, padding=1),
        )

        # 中间监督头（仅训练时使用）
        self.mid_supervision_head = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels // 2, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        input_dtype = x.dtype

        # cuFFT 半精度要求尺寸为 2 的幂，转为 float32 计算 FFT 以兼容任意尺寸
        x = x.float()

        # 2D FFT
        x_fft = torch.fft.fft2(x)
        # 分离幅度和相位
        amp = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        # 幅度 log 压缩，稳定训练
        amp_log = torch.log(amp + 1e-8)

        # 幅度修正
        amp_enhanced = self.amp_net(amp_log)
        # 相位修正
        phase_enhanced = self.phase_net(phase)

        # 还原幅度到线性域
        amp_enhanced_exp = torch.exp(amp_enhanced)

        # 重构复数频谱并 IFFT
        real = amp_enhanced_exp * torch.cos(phase_enhanced)
        imag = amp_enhanced_exp * torch.sin(phase_enhanced)
        x_ifft = torch.fft.ifft2(torch.complex(real, imag)).real

        # 转回原始 dtype
        x_ifft = x_ifft.to(input_dtype)

        # 主特征
        fb = self.output_conv(x_ifft)

        if self.training:
            # 中间监督：生成 3 通道预测图并上采样到原始输入尺寸
            pred_b_mid = self.mid_supervision_head(fb)
            pred_b_mid = F.interpolate(pred_b_mid, size=tuple(self.input_size),
                                       mode='bilinear', align_corners=False)
            return fb, pred_b_mid
        else:
            return fb, None