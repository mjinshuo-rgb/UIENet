"""
评估指标计算
PSNR, SSIM（skimage），UCIQE, UIQM（自实现简化版）
"""

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2hsv


def calculate_psnr(img1, img2, data_range=1.0):
    """img1, img2: numpy arrays (H, W, C), 范围 [0, 1]"""
    return peak_signal_noise_ratio(img1, img2, data_range=data_range)


def calculate_ssim(img1, img2, data_range=1.0):
    """img1, img2: numpy arrays (H, W, C), 范围 [0, 1]"""
    return structural_similarity(img1, img2, channel_axis=-1, data_range=data_range)


def calculate_uciqe(img):
    """
    UCIQE 无参考指标（简化实现）
    基于色度标准差、饱和度和亮度对比度
    img: numpy array (H, W, 3) in [0,1]
    """
    # 色度标准差
    chroma_std = np.std(img[:, :, 0])  # R 通道标准差的近似
    # 饱和度均值（使用 skimage 的 HSV 转换）
    hsv = rgb2hsv(img)
    sat_mean = np.mean(hsv[:, :, 1])
    # 亮度对比度
    luma = img[:, :, 0]  # R 通道近似亮度
    contrast = np.std(luma)
    return 0.4680 * chroma_std + 0.2745 * sat_mean + 0.2576 * contrast


def calculate_uiqm(img):
    """
    UIQM 无参考指标（简化实现）
    基于色彩丰富度、清晰度和对比度
    img: numpy array (H, W, 3) in [0,1]
    """
    img_uint8 = (img * 255).astype(np.uint8)
    # UICM 色彩丰富度
    r, g, b = img_uint8[:, :, 0], img_uint8[:, :, 1], img_uint8[:, :, 2]
    rg = np.abs(r.astype(np.float32) - g.astype(np.float32))
    yb = np.abs(0.5 * (r.astype(np.float32) + g.astype(np.float32)) - b.astype(np.float32))
    uicm = -0.0268 * np.sqrt(np.mean(rg ** 2) + np.mean(yb ** 2))
    # UISM 清晰度
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    from scipy.ndimage import convolve
    grad_x = convolve(luma, sobel_x)
    grad_y = convolve(luma, sobel_y)
    eme = np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2))
    uism = np.log2(1 + eme)
    # UIConM 对比度
    patch_size = 8
    h, w = img_uint8.shape[:2]
    contrast_sum = 0
    patches = 0
    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = luma[i:i+patch_size, j:j+patch_size]
            max_val = np.max(patch)
            min_val = np.min(patch)
            if max_val > 0:
                contrast_sum += (max_val - min_val) / (max_val + min_val)
                patches += 1
    uiconm = contrast_sum / (patches + 1e-8)
    return 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm