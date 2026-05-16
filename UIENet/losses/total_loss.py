"""
总损失组合
整合主损失（L1 + SSIM + VGG + LAB）与辅助损失（Retinex 重建约束、中间监督）。
所有权重从 config.yaml 读取，训练时正确处理中间监督图（推理时为 None）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.perceptual import VGGPerceptualLoss
from losses.color import LABColorLoss


def _gaussian_window(window_size, sigma, channels):
    """创建 2D 高斯窗口用于 SSIM 计算"""
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    window_1d = g.unsqueeze(0) * g.unsqueeze(1)  # outer product
    window = window_1d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    """
    可微分 SSIM 计算（自实现，不依赖 pytorch_ssim）

    Args:
        img1, img2: (B, C, H, W) 张量
        window_size: 高斯窗口尺寸
        sigma: 高斯窗口标准差
    Returns:
        SSIM 均值（标量），值域 [0, 1]
    """
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


class TotalLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambda_l1 = config.loss.lambda_l1
        self.lambda_ssim = config.loss.lambda_ssim
        self.lambda_vgg = config.loss.lambda_vgg
        self.lambda_lab = config.loss.lambda_lab
        self.lambda_retinex = config.loss.lambda_retinex
        self.lambda_d_mid = config.loss.lambda_d_mid
        self.lambda_b_mid = config.loss.lambda_b_mid

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.vgg_loss = VGGPerceptualLoss()
        self.lab_loss = LABColorLoss()

    def forward(self, pred_dict, target, input_img):
        """
        pred_dict: UIENet 输出字典
        target:    真值增强图像 (B,3,H,W)，范围 [-1,1]
        input_img: 原始退化输入 (B,3,H,W)，范围 [-1,1]
        """
        pred_main = pred_dict['output']
        illumination = pred_dict['illumination']
        reflectance = pred_dict['reflectance']
        pred_b_mid = pred_dict['pred_b_mid']
        pred_d_mid = pred_dict['pred_d_mid']

        # 主损失
        loss_l1 = self.l1_loss(pred_main, target)
        loss_ssim = self.ssim_loss(pred_main, target)
        loss_vgg = self.vgg_loss(pred_main, target)
        loss_lab = self.lab_loss(pred_main, target)

        loss_main = (self.lambda_l1 * loss_l1 +
                     self.lambda_ssim * loss_ssim +
                     self.lambda_vgg * loss_vgg +
                     self.lambda_lab * loss_lab)

        # Retinex 重建约束：Illumination × Reflectance ≈ 原始输入
        input_01 = (input_img + 1.0) / 2.0  # 转为 [0,1] 与 Sigmoid 输出对齐
        reconstruction = illumination * reflectance
        loss_retinex = self.l1_loss(reconstruction, input_01) * self.lambda_retinex

        loss_total = loss_main + loss_retinex

        # 中间监督（仅训练时存在）
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
            'retinex': loss_retinex,
            'b_mid': loss_b_mid,
            'd_mid': loss_d_mid,
        }
        return loss_total, loss_dict