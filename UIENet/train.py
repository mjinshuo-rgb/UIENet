"""
训练主脚本 v3.0
- GAN 对抗训练（PatchGAN 判别器）
- 梯度累积等效 batch=8
- Warmup + CosineAnnealing 学习率调度
- EMA（指数移动平均）提升泛化性能
- 频域 + 梯度 + 对抗损失
- TensorBoard 日志
"""

import os
import argparse
import time
import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from utils.config import load_config
from models.uienet import UIENet
from models.discriminator import PatchGANDiscriminator
from losses.total_loss import TotalLoss, discriminator_gan_loss


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


class EMAModel:
    """指数移动平均模型包装器"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """将 EMA 权重写入模型（用于评估）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ---- 数据集 ----
    train_dataset = PairedDataset(config.data.train_dirs, get_transforms(config, train=True))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    print(f"共加载 {len(train_dataset)} 对训练样本")

    # ---- 模型 ----
    model = UIENet(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    # 判别器
    discriminator = PatchGANDiscriminator(in_channels=3, ndf=config.discriminator.ndf,
                                          n_layers=config.discriminator.n_layers).to(device)
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"判别器参数量: {d_params / 1e6:.2f}M")

    # EMA
    ema = EMAModel(model, decay=config.training.ema_decay)

    # ---- 损失函数 ----
    criterion = TotalLoss(config).to(device)

    # ---- 优化器 ----
    optimizer_G = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=config.training.betas
    )
    optimizer_D = optim.AdamW(
        discriminator.parameters(),
        lr=config.training.d_learning_rate,
        weight_decay=config.training.weight_decay,
        betas=config.training.betas
    )

    # 学习率调度器
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_G, T_max=config.training.T_max, eta_min=config.training.eta_min
    )
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_D, T_max=config.training.T_max, eta_min=config.training.eta_min
    )

    # Warmup 函数
    total_steps_per_epoch = len(train_loader) // config.training.gradient_accumulation

    def lr_warmup(current_step):
        warmup_steps = config.training.warmup_epochs * total_steps_per_epoch
        if current_step < warmup_steps:
            return config.training.warmup_start_lr + \
                   (config.training.learning_rate - config.training.warmup_start_lr) * (current_step / warmup_steps)
        return None

    # ---- 日志 ----
    writer = SummaryWriter(log_dir=config.logging.log_dir)
    checkpoint_dir = Path(config.logging.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    scaler_G = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    scaler_D = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    gan_start_epoch = config.training.gan_start_epoch

    for epoch in range(config.training.num_epochs):
        model.train()
        discriminator.train()
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        epoch_start = time.time()
        use_gan = (epoch >= gan_start_epoch)

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}")
        for batch_idx, (input_img, target) in enumerate(progress):
            input_img = input_img.to(device)
            target = target.to(device)

            # ============================
            # 1. 训练生成器
            # ============================
            with torch.amp.autocast('cuda', enabled=(scaler_G is not None)):
                pred_dict = model(input_img)
                loss_total, loss_dict = criterion(pred_dict, target, input_img,
                                                  discriminator=discriminator,
                                                  use_gan=use_gan)
                loss_total = loss_total / config.training.gradient_accumulation

            if scaler_G is not None:
                scaler_G.scale(loss_total).backward()
            else:
                loss_total.backward()

            # ============================
            # 2. 训练判别器
            # ============================
            if use_gan:
                d_update = (batch_idx % config.training.d_update_freq == 0)
                if d_update:
                    with torch.amp.autocast('cuda', enabled=(scaler_D is not None)):
                        fake_img = pred_dict['output'].detach()
                        loss_D = discriminator_gan_loss(discriminator, target, fake_img)
                        loss_D = loss_D / config.training.gradient_accumulation

                    if scaler_D is not None:
                        scaler_D.scale(loss_D).backward()
                    else:
                        loss_D.backward()
                else:
                    loss_D = torch.tensor(0.0)
            else:
                loss_D = torch.tensor(0.0)

            # ============================
            # 3. 梯度累积更新
            # ============================
            if (batch_idx + 1) % config.training.gradient_accumulation == 0:
                # 更新生成器
                if scaler_G is not None:
                    scaler_G.step(optimizer_G)
                    scaler_G.update()
                else:
                    optimizer_G.step()
                optimizer_G.zero_grad()

                # 更新判别器
                if use_gan and loss_D.item() != 0:
                    if scaler_D is not None:
                        scaler_D.step(optimizer_D)
                        scaler_D.update()
                    else:
                        optimizer_D.step()
                    optimizer_D.zero_grad()

                # EMA 更新
                ema.update()

                global_step += 1

                # Warmup
                warmup_lr = lr_warmup(global_step)
                if warmup_lr is not None:
                    for param_group in optimizer_G.param_groups:
                        param_group['lr'] = warmup_lr

            # 记录
            epoch_loss_G += loss_total.item() * config.training.gradient_accumulation
            if use_gan:
                epoch_loss_D += (loss_D.item() if loss_D.item() != 0 else 0) * config.training.gradient_accumulation

            postfix_dict = {'G': f"{loss_dict['total'].item():.3f}"}
            if use_gan and loss_D.item() != 0:
                postfix_dict['D'] = f"{loss_D.item():.3f}"
            progress.set_postfix(postfix_dict)

            # TensorBoard（每 10 步记录）
            if global_step % 10 == 0 and global_step > 0:
                for k, v in loss_dict.items():
                    writer.add_scalar(f'Loss/{k}', v.item(), global_step)
                if use_gan and loss_D.item() != 0:
                    writer.add_scalar('Loss/D', loss_D.item(), global_step)

        # ---- Epoch 结束 ----
        if epoch >= config.training.warmup_epochs:
            scheduler_G.step()
            if use_gan:
                scheduler_D.step()

        avg_loss_G = epoch_loss_G / len(train_loader)
        writer.add_scalar('Epoch/loss_G', avg_loss_G, epoch)
        writer.add_scalar('Epoch/lr_G', optimizer_G.param_groups[0]['lr'], epoch)
        if use_gan:
            writer.add_scalar('Epoch/loss_D', epoch_loss_D / len(train_loader), epoch)

        print(f"Epoch {epoch+1}: G_loss={avg_loss_G:.4f}, D_loss={epoch_loss_D/len(train_loader):.4f}, "
              f"时间={time.time()-epoch_start:.1f}s, lr={optimizer_G.param_groups[0]['lr']:.2e}")

        # ---- 保存检查点 ----
        if (epoch + 1) % config.logging.save_every == 0:
            # 临时应用 EMA 权重保存
            ema_backup = copy.deepcopy(model.state_dict())
            ema.apply_shadow()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'scheduler_D_state_dict': scheduler_D.state_dict(),
                'config': config,
            }, checkpoint_dir / f'uienet_epoch{epoch+1}.pth')
            model.load_state_dict(ema_backup)  # 恢复训练权重
            print(f"  检查点已保存: uienet_epoch{epoch+1}.pth")

    writer.close()
    print("训练完成。")


if __name__ == '__main__':
    main()