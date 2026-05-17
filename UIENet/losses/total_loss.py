"""
总损失组合 v3.0
整合：
  - 像素损失：L1 + SSIM
  - 感知损失：VGG Perceptual
  - 色彩损失：LAB ab 通道
  - 对抗损失：LSGAN (PatchGAN)
  - 结构损失：梯度损失（边缘锐度）+ 频域损失（纹理匹配）
  - 辅助损失：分支 B/D 中间监督
所有权重从 config.yaml 读取。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.perceptual import VGGPerceptualLoss
from losses.color import LABColorLoss


# ---------------------------------------------------------------------------
# SSIM（自实现，无第三方依赖）
# ---------------------------------------------------------------------------

def _gaussian_window(window_size, sigma, channels):
    """创建 2D 高斯窗口用于 SSIM 计算"""
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    window_1d = g.unsqueeze(0) * g.unsqueeze(1)
    window = window_1d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    """可微分 SSIM，返回均值标量（值域 [0,1]）"""
    channels = img1.size(1)
    window = _gaussian_window(window_size, sigma, channels).to(img1.device).to(img1.dtype)
    padding = window_size // 2

    mu1 = F.conv2d(img1, window, padding=padding, groups=channels)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channels)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, pred, target):
        return 1.0 - _ssim(pred, target, self.window_size, self.sigma)


# ---------------------------------------------------------------------------
# 梯度损失（边缘锐度）
# ---------------------------------------------------------------------------

def gradient_loss(pred, target):
    """
    计算图像梯度的 L1 损失，鼓励边缘对齐。
    使用 Sobel 类算子近似 ∂/∂x 和 ∂/∂y。
    """
    def sobel_kernels(device, dtype):
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                          device=device, dtype=dtype).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                          device=device, dtype=dtype).view(1, 1, 3, 3)
        return kx, ky

    B, C, H, W = pred.shape
    kx, ky = sobel_kernels(pred.device, pred.dtype)
    kx = kx.repeat(C, 1, 1, 1)
    ky = ky.repeat(C, 1, 1, 1)

    pred_gx = F.conv2d(pred, kx, padding=1, groups=C)
    pred_gy = F.conv2d(pred, ky, padding=1, groups=C)
    target_gx = F.conv2d(target, kx, padding=1, groups=C)
    target_gy = F.conv2d(target, ky, padding=1, groups=C)

    return F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)


# ---------------------------------------------------------------------------
# 频域损失（纹理匹配）
# ---------------------------------------------------------------------------

def frequency_loss(pred, target):
    """
    FFT 幅度谱 L1 损失，约束频域纹理一致性。
    """
    pred_f = torch.fft.fft2(pred.float())
    target_f = torch.fft.fft2(target.float())
    return F.l1_loss(torch.abs(pred_f), torch.abs(target_f))


# ---------------------------------------------------------------------------
# GAN 损失（LSGAN）
# ---------------------------------------------------------------------------

def generator_gan_loss(discriminator, fake_img):
    """
    LSGAN 生成器损失: E[(D(G(x)) - 1)^2]
    discriminator: PatchGAN 判别器
    fake_img: 生成图像
    """
    pred_fake = discriminator(fake_img)
    target_real = torch.ones_like(pred_fake) * 0.9  # label smoothing
    return F.mse_loss(pred_fake, target_real)


def discriminator_gan_loss(discriminator, real_img, fake_img):
    """
    LSGAN 判别器损失: E[(D(real) - 1)^2] + E[(D(fake))^2]
    """
    pred_real = discriminator(real_img)
    pred_fake = discriminator(fake_img.detach())

    target_real = torch.ones_like(pred_real) * 0.9
    target_fake = torch.zeros_like(pred_fake)

    loss_real = F.mse_loss(pred_real, target_real)
    loss_fake = F.mse_loss(pred_fake, target_fake)
    return (loss_real + loss_fake) * 0.5


# ---------------------------------------------------------------------------
# 总损失类
# ---------------------------------------------------------------------------

class TotalLoss(nn.Module):
    """生成器总损失（不含对抗损失，对抗损失在 train.py 中动态计算）"""
    def __init__(self, config):
        super().__init__()
        self.lambda_l1 = config.loss.lambda_l1
        self.lambda_ssim = config.loss.lambda_ssim
        self.lambda_vgg = config.loss.lambda_vgg
        self.lambda_lab = config.loss.lambda_lab
        self.lambda_gan = config.loss.lambda_gan
        self.lambda_gradient = config.loss.lambda_gradient
        self.lambda_frequency = config.loss.lambda_frequency
        self.lambda_retinex = config.loss.lambda_retinex
        self.lambda_d_mid = config.loss.lambda_d_mid
        self.lambda_b_mid = config.loss.lambda_b_mid

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.vgg_loss = VGGPerceptualLoss()
        self.lab_loss = LABColorLoss()

    def forward(self, pred_dict, target, input_img, discriminator=None, use_gan=False):
        """
        Args:
            pred_dict:     UIENet 输出字典
            target:        真值增强图像 (B,3,H,W), [-1,1]
            input_img:     原始退化图像 (B,3,H,W), [-1,1]
            discriminator: PatchGAN 判别器（可选）
            use_gan:       是否启用对抗损失
        Returns:
            loss_total: 总损失标量
            loss_dict:  各分量损失字典
        """
        pred_main = pred_dict['output']
        pred_b_mid = pred_dict['pred_b_mid']
        pred_d_mid = pred_dict['pred_d_mid']

        # ---- 主损失 ----
        loss_l1 = self.l1_loss(pred_main, target)
        loss_ssim = self.ssim_loss(pred_main, target)
        loss_vgg = self.vgg_loss(pred_main, target)
        loss_lab = self.lab_loss(pred_main, target)

        loss_main = (self.lambda_l1 * loss_l1 +
                     self.lambda_ssim * loss_ssim +
                     self.lambda_vgg * loss_vgg +
                     self.lambda_lab * loss_lab)

        # ---- 梯度损失 ----
        loss_grad = gradient_loss(pred_main, target) * self.lambda_gradient

        # ---- 频域损失 ----
        loss_freq = frequency_loss(pred_main, target) * self.lambda_frequency

        loss_total = loss_main + loss_grad + loss_freq

        # ---- 对抗损失 ----
        if use_gan and discriminator is not None:
            loss_gan = generator_gan_loss(discriminator, pred_main) * self.lambda_gan
            loss_total = loss_total + loss_gan
        else:
            loss_gan = torch.tensor(0.0, device=loss_total.device)

        # ---- 中间监督 ----
        if pred_b_mid is not None:
            loss_b_mid = self.l1_loss(pred_b_mid, target) * self.lambda_b_mid
            loss_total = loss_total + loss_b_mid
        else:
            loss_b_mid = torch.tensor(0.0, device=loss_total.device)

        if pred_d_mid is not None:
            loss_d_mid = self.l1_loss(pred_d_mid, target) * self.lambda_d_mid
            loss_total = loss_total + loss_d_mid
        else:
            loss_d_mid = torch.tensor(0.0, device=loss_total.device)

        loss_dict = {
            'total': loss_total,
            'l1': loss_l1,
            'ssim': loss_ssim,
            'vgg': loss_vgg,
            'lab': loss_lab,
            'gradient': loss_grad,
            'frequency': loss_freq,
            'gan': loss_gan,
            'retinex': torch.tensor(0.0, device=loss_total.device),
            'b_mid': loss_b_mid,
            'd_mid': loss_d_mid,
        }
        return loss_total, loss_dict