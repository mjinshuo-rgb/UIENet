"""
可视化工具：训练过程对比图保存
"""

import torch
from torchvision.utils import make_grid
import numpy as np
from PIL import Image


def tensor_to_image(tensor):
    """将 [-1,1] 的 Tensor 转换为 [0,255] 的 numpy array"""
    img = tensor.detach().cpu().float().numpy()
    img = (img + 1.0) / 2.0  # [-1,1] -> [0,1]
    img = img.clip(0, 1)
    img = img.transpose(1, 2, 0)  # C,H,W -> H,W,C
    img = (img * 255).astype(np.uint8)
    return img


def save_comparison(input_img, pred_img, target_img, save_path):
    """
    保存增强前后对比图（水平拼接：输入 | 增强结果 | 真值）
    input_img:   (3,H,W) Tensor, [-1,1]
    pred_img:    (3,H,W) Tensor, [-1,1]
    target_img:  (3,H,W) Tensor, [-1,1]
    """
    input_np = tensor_to_image(input_img)
    pred_np = tensor_to_image(pred_img)
    target_np = tensor_to_image(target_img)
    comparison = np.concatenate([input_np, pred_np, target_np], axis=1)
    Image.fromarray(comparison).save(save_path)


def log_images_to_tensorboard(writer, input_img, pred_img, target_img, global_step, tag='Train'):
    """将对比图写入 TensorBoard"""
    input_np = tensor_to_image(input_img)
    pred_np = tensor_to_image(pred_img)
    target_np = tensor_to_image(target_img)
    comparison = np.concatenate([input_np, pred_np, target_np], axis=1)
    # TensorBoard 需要 (C,H,W) 格式
    comparison_tensor = torch.from_numpy(comparison.transpose(2, 0, 1))
    writer.add_image(f'{tag}/Comparison', comparison_tensor, global_step)