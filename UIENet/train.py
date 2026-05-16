"""
训练主脚本
- 使用 config.yaml 配置所有超参数
- 梯度累积等效 batch=8
- Warmup + CosineAnnealing 学习率调度
- TensorBoard 日志
"""

import os
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from utils.config import load_config
from models.uienet import UIENet
from losses.total_loss import TotalLoss
from data.datasets import PairedDataset


def get_transforms(config, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize([int(s * 1.15) for s in config.data.train_size]),
            transforms.RandomCrop(config.data.train_size),
            transforms.RandomHorizontalFlip(p=config.data.horizontal_flip_prob),
            transforms.RandomVerticalFlip(p=config.data.vertical_flip_prob),
            transforms.ColorJitter(
                brightness=config.data.color_jitter.brightness,
                contrast=config.data.color_jitter.contrast,
                saturation=config.data.color_jitter.saturation,
                hue=config.data.color_jitter.hue
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.data.normalize_mean, std=config.data.normalize_std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(config.data.train_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.data.normalize_mean, std=config.data.normalize_std)
        ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据集
    train_dataset = PairedDataset(config.data.train_dirs, get_transforms(config, train=True))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # 模型
    model = UIENet(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    # 损失函数
    criterion = TotalLoss(config).to(device)

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=config.training.betas
    )

    # 学习率调度器（Cosine Annealing）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.T_max,
        eta_min=config.training.eta_min
    )

    # Warmup 函数
    def lr_warmup(current_step):
        warmup_steps = config.training.warmup_epochs * len(train_loader) // config.training.gradient_accumulation
        if current_step < warmup_steps:
            return config.training.warmup_start_lr + \
                   (config.training.learning_rate - config.training.warmup_start_lr) * (current_step / warmup_steps)
        return None  # 由 scheduler 接管

    # 日志
    writer = SummaryWriter(log_dir=config.logging.log_dir)
    checkpoint_dir = Path(config.logging.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    for epoch in range(config.training.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        optimizer.zero_grad()

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}")
        for batch_idx, (input_img, target) in enumerate(progress):
            input_img = input_img.to(device)
            target = target.to(device)

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                pred_dict = model(input_img)
                loss_total, loss_dict = criterion(pred_dict, target, input_img)
                # 梯度累积：除以累积步数
                loss_total = loss_total / config.training.gradient_accumulation

            if scaler is not None:
                scaler.scale(loss_total).backward()
            else:
                loss_total.backward()

            # 更新步数
            if (batch_idx + 1) % config.training.gradient_accumulation == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Warmup 调整学习率
                warmup_lr = lr_warmup(global_step)
                if warmup_lr is not None:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr

            # 记录 batch 损失
            epoch_loss += loss_total.item() * config.training.gradient_accumulation
            progress.set_postfix(loss=loss_dict['total'].item())

            if global_step % 10 == 0:
                for k, v in loss_dict.items():
                    writer.add_scalar(f'Loss/{k}', v.item(), global_step)

        # 每个 epoch 结束，更新学习率（warmup 结束后由 scheduler 接管）
        # 修改为：
        if epoch >= config.training.warmup_epochs and epoch > 0:
            scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Epoch/loss', avg_loss, epoch)
        writer.add_scalar('Epoch/lr', optimizer.param_groups[0]['lr'], epoch)
        print(f"Epoch {epoch+1}: 平均损失 {avg_loss:.4f}, 时间 {time.time()-epoch_start:.1f}s, "
              f"学习率 {optimizer.param_groups[0]['lr']:.2e}")

        # 保存检查点
        if (epoch + 1) % config.logging.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_dir / f'uienet_epoch{epoch+1}.pth')

    writer.close()
    print("训练完成。")


if __name__ == '__main__':
    main()