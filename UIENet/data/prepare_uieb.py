"""
UIEB 数据集质量筛选脚本
使用亮度方差 + 色彩饱和度作为质量代理指标（近似NIQE），
筛选高质量参考图，保留约650对样本存入 train_filtered 目录。
"""

import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def quality_score(img_path):
    """
    质量评分：亮度方差 + 饱和度均值（越高越好）
    作为 NIQE 的轻量代替，无需额外模型
    """
    img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0
    # 亮度方差（反映对比度）
    luma = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    luma_var = np.var(luma)
    # 饱和度均值
    max_c = img.max(axis=2)
    min_c = img.min(axis=2)
    sat = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-8), 0)
    sat_mean = np.mean(sat)
    return luma_var * 0.6 + sat_mean * 0.4


def filter_uieb(source_dir, output_dir, keep_ratio=0.75):
    """
    source_dir: UIEB 原始目录，包含 input/ 和 gt/ 子目录
    output_dir: 筛选后输出目录（train_filtered）
    keep_ratio: 保留比例，0.75 约保留 650 对（原始890对）
    """
    gt_dir = Path(source_dir) / 'gt'
    input_dir = Path(source_dir) / 'input'
    out_gt = Path(output_dir) / 'gt'
    out_input = Path(output_dir) / 'input'
    out_gt.mkdir(parents=True, exist_ok=True)
    out_input.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(gt_dir.glob('*.[jp][pn]g'))
    print(f"原始样本数: {len(gt_files)}")

    # 计算所有 GT 图的质量分数
    scores = []
    for gt_path in tqdm(gt_files, desc="计算质量分数"):
        score = quality_score(str(gt_path))
        scores.append((score, gt_path))

    # 按分数降序排列，保留前 keep_ratio
    scores.sort(key=lambda x: x[0], reverse=True)
    keep_n = int(len(scores) * keep_ratio)
    selected = scores[:keep_n]
    print(f"保留样本数: {keep_n}")

    # 复制筛选后的文件
    for score, gt_path in tqdm(selected, desc="复制文件"):
        fname = gt_path.name
        input_path = input_dir / fname
        if input_path.exists():
            shutil.copy(gt_path, out_gt / fname)
            shutil.copy(input_path, out_input / fname)

    print(f"筛选完成，输出目录: {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True,
                        help='UIEB 原始目录（含 input/ 和 gt/ 子目录）')
    parser.add_argument('--output', type=str,
                        default='./datasets/UIEB/train_filtered',
                        help='筛选后的输出目录')
    parser.add_argument('--keep_ratio', type=float, default=0.75)
    args = parser.parse_args()
    filter_uieb(args.source, args.output, args.keep_ratio)
