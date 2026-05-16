"""
PatchGAN 判别器（70×70）

用于对抗训练，提升增强图像的感知质量。
- 5 层 CNN，感受野 70×70
- Spectral Normalization 保证训练稳定性
- 输出 (B, 1, H//16, W//16) 的逐块真假概率图

参考: pix2pix / CycleGAN
"""

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN 判别器

    Args:
        in_channels: 输入通道数（RGB = 3 for real/fake 拼接 = 6）
        ndf: 第一层卷积通道数（默认 64）
        n_layers: 卷积层数（默认 3，即总共 5 层：C64→C128→C256→C512→1）
    """
    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf_mult = 1
        for n in range(1, n_layers + 1):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                        kernel_size=4, stride=2, padding=1, bias=False)),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        # 最后一层：输出 1 通道逐块真假概率
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                    kernel_size=4, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: 输入图像 (B, C, H, W)
        Returns:
            逐块真假概率图 (B, 1, H//16, W//16)
        """
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    多尺度判别器（两个尺度）
    尺度1: 原图 → 70×70 PatchGAN
    尺度2: 2× 下采样 → 70×70 PatchGAN

    参考: pix2pixHD
    """
    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super().__init__()
        self.d1 = PatchGANDiscriminator(in_channels, ndf, n_layers)
        self.d2 = PatchGANDiscriminator(in_channels, ndf, n_layers)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x):
        out1 = self.d1(x)
        out2 = self.d2(self.downsample(x))
        return [out1, out2]
