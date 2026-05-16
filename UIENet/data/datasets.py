"""
数据集加载模块
支持从多个训练目录加载配对图像，以及多个测试集。
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class PairedDataset(Dataset):
    """配对水下图像数据集（退化图 - 真值图）"""
    def __init__(self, data_dirs, transform=None):
        self.samples = []
        self.transform = transform
        for d in data_dirs:
            input_dir = os.path.join(d, 'input')
            gt_dir = os.path.join(d, 'gt')
            if not os.path.exists(input_dir) or not os.path.exists(gt_dir):
                print(f"警告：跳过缺失目录 {d}")
                continue
            input_files = sorted(os.listdir(input_dir))
            for fname in input_files:
                input_path = os.path.join(input_dir, fname)
                gt_path = os.path.join(gt_dir, fname)
                if os.path.exists(gt_path):
                    self.samples.append((input_path, gt_path))
        print(f"共加载 {len(self.samples)} 对训练样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, gt_path = self.samples[idx]
        input_img = Image.open(input_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        # 同步增强：将两张图左右拼接后统一 transform
        combined = Image.new('RGB', (input_img.width * 2, input_img.height))
        combined.paste(input_img, (0, 0))
        combined.paste(gt_img, (input_img.width, 0))
        if self.transform:
            combined = self.transform(combined)
            w = combined.shape[2] // 2
            input_tensor = combined[:, :, :w]
            gt_tensor = combined[:, :, w:]
        return input_tensor, gt_tensor


class UnpairedDataset(Dataset):
    """无参考测试集（仅退化图）"""
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.image_paths = sorted(Path(data_dir).glob('*.[jp][pn]g'))
        if len(self.image_paths) == 0:
            self.image_paths = sorted(Path(data_dir).glob('*.bmp'))
        print(f"共加载 {len(self.image_paths)} 张测试图像")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, str(path.name)