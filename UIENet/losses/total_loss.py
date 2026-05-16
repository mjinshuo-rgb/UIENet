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

try:
    from pytorch_ssim import ssim
except ImportError:
    ssim = None


class SSIMLoss(nn.Module):
    def forward(self, pred, target):
        if ssim is None:
            raise ImportError("pytorch_ssim 未安装，请 pip install pytorch_ssim")
        return 1.0 - ssim(pred, target)


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