"""
VGG 感知损失
使用预训练 VGG19 的 conv3_3 层特征计算 L1 距离，
鼓励增强图像与真值在高层语义上的一致性。
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.slice = nn.Sequential()
        # 取前10层（conv3_3 之后）
        for i in range(10):
            self.slice.add_module(str(i), vgg[i])
        for param in self.slice.parameters():
            param.requires_grad = False

        # VGG 归一化参数（ImageNet 统计值）
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def forward(self, pred, target):
        # 输入范围 [-1, 1] → [0, 1] 再归一化
        pred = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0
        pred_feat = self.slice(self.normalize(pred))
        target_feat = self.slice(self.normalize(target))
        return torch.nn.functional.l1_loss(pred_feat, target_feat)