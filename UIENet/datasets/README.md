# 数据集目录结构说明

请按以下结构组织数据集，与 config.yaml 中的路径保持一致。

## 目录结构

datasets/
├── EUVP/
│   ├── train/
│   │   ├── input/     # 退化水下图像
│   │   └── gt/        # 对应真值图像
│   └── test/
│       ├── input/
│       └── gt/
├── UIEB/
│   ├── raw/           # UIEB 原始数据（890对）
│   │   ├── input/
│   │   └── gt/
│   └── train_filtered/ # 筛选后数据（运行 prepare_uieb.py 生成）
│       ├── input/
│       └── gt/
├── LSUI/
│   └── test/
│       ├── input/
│       └── gt/
└── U90/               # 无参考测试集，直接放图片，无子目录

## 数据集来源

- EUVP: https://irvlab.cs.umn.edu/resources/euvp-dataset
- UIEB: https://li-chongyi.github.io/proj_benchmark.html
- LSUI: https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement
- U90: 随 UIEB 论文附带

## UIEB 筛选步骤

数据下载后，运行以下命令筛选高质量样本：

    python data/prepare_uieb.py \
        --source ./datasets/UIEB/raw \
        --output ./datasets/UIEB/train_filtered \
        --keep_ratio 0.75

预计保留约 650 对样本。
