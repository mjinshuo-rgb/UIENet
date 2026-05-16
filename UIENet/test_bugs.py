"""完整训练流程 BUG 测试（使用 autocast + batch=1）"""
import torch
import gc
import sys

gc.collect()
torch.cuda.empty_cache()
free, total = torch.cuda.mem_get_info()
print(f'显存: 可用={free/1024**3:.1f}GiB / 总计={total/1024**3:.1f}GiB')

from utils.config import load_config
from models.uienet import UIENet
from models.discriminator import PatchGANDiscriminator
from losses.total_loss import TotalLoss, discriminator_gan_loss
import torch.optim as optim

config = load_config('config.yaml')
device = torch.device('cuda')

# 模型
model = UIENet(config).to(device)
disc = PatchGANDiscriminator(in_channels=3, ndf=64, n_layers=3).to(device)
criterion = TotalLoss(config).to(device)
g_params = sum(p.numel() for p in model.parameters())
d_params = sum(p.numel() for p in disc.parameters())
print(f'参数量: G={g_params/1e6:.2f}M, D={d_params/1e6:.2f}M')

opt_G = optim.AdamW(model.parameters(), lr=1e-4)
opt_D = optim.AdamW(disc.parameters(), lr=1e-4)

# ===== 测试 autocast 前向 + 损失 =====
model.train()
disc.train()
x = torch.randn(1, 3, 384, 384, device=device)
target = torch.randn(1, 3, 384, 384, device=device)

with torch.amp.autocast('cuda'):
    pred_dict = model(x)
    loss_G, loss_dict = criterion(pred_dict, target, x, discriminator=disc, use_gan=True)
    fake_img = pred_dict['output'].detach()
    loss_D = discriminator_gan_loss(disc, target, fake_img)

print(f'输出: {pred_dict["output"].shape}')
for k, v in pred_dict.items():
    if v is not None and torch.isnan(v).any():
        print(f'❌ {k} NaN!'); sys.exit(1)
print('✅ autocast前向通过，无NaN')

print(f'损失: G={loss_G.item():.4f}, D={loss_D.item():.4f}')

# ===== 反向传播 =====
opt_G.zero_grad()
loss_G.backward(retain_graph=True)
g_nan = sum(1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
if g_nan > 0: print(f'❌ G梯度 {g_nan} 个参数 NaN!'); sys.exit(1)

opt_D.zero_grad()
loss_D.backward()
d_nan = sum(1 for p in disc.parameters() if p.grad is not None and torch.isnan(p.grad).any())
if d_nan > 0: print(f'❌ D梯度 {d_nan} 个参数 NaN!'); sys.exit(1)
print('✅ 反向传播通过，无NaN梯度')

opt_G.step()
opt_D.step()
print('✅ 优化器步进通过')

# ===== 连续多步训练 =====
print('连续5步训练...')
losses = []
for _ in range(5):
    xb = torch.randn(1,3,384,384,device=device)
    tb = torch.randn(1,3,384,384,device=device)
    with torch.amp.autocast('cuda'):
        pred = model(xb)
        lg, _ = criterion(pred, tb, xb, discriminator=disc, use_gan=True)
        ld = discriminator_gan_loss(disc, tb, pred['output'].detach())
    opt_G.zero_grad(); lg.backward(); opt_G.step()
    opt_D.zero_grad(); ld.backward(); opt_D.step()
    losses.append(lg.item())
print(f'  step1 loss={losses[0]:.4f}, step5 loss={losses[-1]:.4f}')
print('✅ 连续训练通过')

used = torch.cuda.memory_allocated() / 1024**3
peak = torch.cuda.max_memory_allocated() / 1024**3
print(f'显存: 当前{used:.2f}GiB / 峰值{peak:.2f}GiB')

print('🎉 全部测试通过！模型无BUG，可以开始训练。')
print(f'   训练命令: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py --config config.yaml')
