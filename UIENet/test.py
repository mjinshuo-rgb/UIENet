"""
测试评估脚本
在三个测试集（EUVP, LSUI, U90）上计算指标并保存增强结果。
"""

import argparse
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.config import load_config
from utils.metrics import calculate_psnr, calculate_ssim, calculate_uciqe, calculate_uiqm
from models.uienet import UIENet
from data.datasets import PairedDataset, UnpairedDataset


def evaluate(config_path, checkpoint_path, output_dir):
    config = load_config(config_path)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    # 测试集 transform（统一 Resize 到训练尺寸，不进行数据增强）
    test_transform = transforms.Compose([
        transforms.Resize(config.data.train_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.data.normalize_mean, std=config.data.normalize_std)
    ])

    # 加载模型
    model = UIENet(config).to(device)
    model.eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    results = {}

    # 1. EUVP 测试集（有参考）
    if config.data.test_dirs.euvp:
        print("\n===== 评估 EUVP =====")
        euvp_dataset = PairedDataset([config.data.test_dirs.euvp], transform=test_transform)
        euvp_loader = DataLoader(euvp_dataset, batch_size=1, shuffle=False, num_workers=2)
        psnr_list, ssim_list, uciqe_list, uiqm_list = [], [], [], []
        with torch.no_grad():
            for input_img, target in tqdm(euvp_loader, desc="EUVP"):
                input_img, target = input_img.to(device), target.to(device)
                pred = model(input_img)['output']
                pred_np = tensor_to_numpy(pred[0])
                target_np = tensor_to_numpy(target[0])
                psnr_list.append(calculate_psnr(pred_np, target_np))
                ssim_list.append(calculate_ssim(pred_np, target_np))
                uciqe_list.append(calculate_uciqe(pred_np))
                uiqm_list.append(calculate_uiqm(pred_np))
        results['EUVP'] = {
            'PSNR': np.mean(psnr_list), 'SSIM': np.mean(ssim_list),
            'UCIQE': np.mean(uciqe_list), 'UIQM': np.mean(uiqm_list)
        }

    # 2. LSUI 测试集（有参考）
    if config.data.test_dirs.lsui:
        print("\n===== 评估 LSUI =====")
        lsui_dataset = PairedDataset([config.data.test_dirs.lsui], transform=test_transform)
        lsui_loader = DataLoader(lsui_dataset, batch_size=1, shuffle=False, num_workers=2)
        psnr_list, ssim_list, uciqe_list, uiqm_list = [], [], [], []
        with torch.no_grad():
            for input_img, target in tqdm(lsui_loader, desc="LSUI"):
                input_img, target = input_img.to(device), target.to(device)
                pred = model(input_img)['output']
                pred_np = tensor_to_numpy(pred[0])
                target_np = tensor_to_numpy(target[0])
                psnr_list.append(calculate_psnr(pred_np, target_np))
                ssim_list.append(calculate_ssim(pred_np, target_np))
                uciqe_list.append(calculate_uciqe(pred_np))
                uiqm_list.append(calculate_uiqm(pred_np))
        results['LSUI'] = {
            'PSNR': np.mean(psnr_list), 'SSIM': np.mean(ssim_list),
            'UCIQE': np.mean(uciqe_list), 'UIQM': np.mean(uiqm_list)
        }

    # 3. U90 测试集（无参考）
    if config.data.test_dirs.u90:
        print("\n===== 评估 U90 =====")
        u90_dataset = UnpairedDataset(config.data.test_dirs.u90, transform=test_transform)
        u90_loader = DataLoader(u90_dataset, batch_size=1, shuffle=False, num_workers=2)
        uciqe_list, uiqm_list = [], []
        os.makedirs(os.path.join(output_dir, 'U90_enhanced'), exist_ok=True)
        with torch.no_grad():
            for img, name in tqdm(u90_loader, desc="U90"):
                img = img.to(device)
                pred = model(img)['output']
                pred_np = tensor_to_numpy(pred[0])
                uciqe_list.append(calculate_uciqe(pred_np))
                uiqm_list.append(calculate_uiqm(pred_np))
                # 保存增强图
                save_path = os.path.join(output_dir, 'U90_enhanced', name[0])
                Image.fromarray((pred_np * 255).astype(np.uint8)).save(save_path)
        results['U90'] = {'UCIQE': np.mean(uciqe_list), 'UIQM': np.mean(uiqm_list)}

    # 输出结果
    print("\n========== 评估结果汇总 ==========")
    for dataset, metrics in results.items():
        print(f"{dataset}: ", end="")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}  ", end="")
        print()

    # 保存为文本文件
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write("========== 评估结果 ==========\n")
        for dataset, metrics in results.items():
            f.write(f"{dataset}: ")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}  ")
            f.write("\n")
    print(f"结果已保存到 {output_dir}/results.txt")


def tensor_to_numpy(tensor):
    """[-1,1] Tensor -> [0,1] numpy (H,W,C)"""
    img = tensor.cpu().float().numpy()
    img = (img + 1.0) / 2.0
    img = img.clip(0, 1)
    img = img.transpose(1, 2, 0)
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重路径')
    parser.add_argument('--output_dir', type=str, default='./test_results')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(args.config, args.checkpoint, args.output_dir)