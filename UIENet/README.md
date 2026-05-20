# UIENet: Underwater Image Enhancement Network

UIENet 是一个基于多阶段物理先验和深度学习的水下图像增强网络，旨在提升水下图像的质量，达到 28dB+ 的 PSNR 性能。

## 项目架构

本项目采用三阶段架构：

### Phase 1: 深度物理先验
- 基于 Retinex 理论的光照-反射分解
- 暗通道先验（Dark Channel Prior）
- 浑浊度估计（Turbidity Estimation）

### Phase 2: 多域特征协同
- **Swin Transformer 分支**：捕捉全局上下文信息
- **频域分支**：处理频率域特征
- **CBAM 注意力分支**：通道和空间注意力机制
- **UNet 分支**：多尺度特征融合

### Phase 3: 融合与 LAB 校正
- 多分支特征融合
- LAB 颜色空间校正
- PatchGAN 判别器进行对抗训练

## 目录结构

```
UIENet/
├── config.yaml          # 模型和训练配置
├── train.py             # 训练主脚本
├── test.py              # 测试脚本
├── test_bugs.py         # 测试调试脚本
├── requirements.txt     # Python 依赖
├── data/                # 数据加载和预处理
│   ├── __init__.py
│   ├── datasets.py      # 数据集类
│   └── prepare_uieb.py  # UIEB 数据集准备脚本
├── models/              # 模型定义
│   ├── uienet.py        # UIENet 主模型
│   ├── discriminator.py # PatchGAN 判别器
│   └── ...              # 其他模型组件
├── losses/              # 损失函数
│   └── total_loss.py    # 综合损失函数
├── utils/               # 工具函数
│   └── config.py        # 配置加载工具
└── datasets/            # 数据集目录（不上传到 Git）
    ├── EUVP/
    ├── LSUI/
    ├── U90/
    └── UIEB/
```

## 安装

1. 克隆仓库：
```bash
git clone git@github.com:mjinshuo-rgb/UIENet.git
cd UIENet
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据集准备

支持以下数据集：
- **EUVP**：水下图像增强数据集
- **LSUI**：大规模水下图像数据集
- **U90**：水下图像数据集
- **UIEB**：水下图像增强基准数据集

数据集应放置在 `datasets/` 目录下，结构如下：
```
datasets/
├── EUVP/
│   ├── train/
│   │   ├── input/    # 输入图像
│   │   └── gt/       # 真值图像
│   └── test/
│       ├── input/
│       └── gt/
├── LSUI/
├── U90/
└── UIEB/
    └── raw/
        ├── input/
        └── gt/
```

对于 UIEB 数据集，可以使用 `data/prepare_uieb.py` 脚本进行准备。

## 训练

1. 修改配置文件 `config.yaml` 以适应你的数据集和硬件环境。

2. 开始训练：
```bash
python train.py --config config.yaml
```

3. 训练过程中会保存以下内容：
   - 模型权重（checkpoints/）
   - 训练日志（logs/）
   - TensorBoard 日志

4. 使用 TensorBoard 监控训练：
```bash
tensorboard --logdir logs/
```

## 测试

1. 使用训练好的模型进行测试：
```bash
python test.py --config config.yaml --checkpoint checkpoints/best_model.pth
```

2. 测试结果将保存在 `test_results/` 目录中。

## 模型配置

所有超参数都在 `config.yaml` 中配置，包括：

- **Phase 1 参数**：Retinex 中间通道数、密度估计通道数等
- **Phase 2 参数**：Swin Transformer 配置、频域分支参数、CBAM 参数等
- **Phase 3 参数**：融合维度、注意力头数、LAB 校正开关等
- **判别器参数**：判别器层数、R1 梯度惩罚系数等
- **数据参数**：图像大小、数据增强参数、归一化参数等
- **训练参数**：学习率、批大小、训练轮数等

## 性能指标

- **PSNR**：峰值信噪比（目标 28dB+）
- **SSIM**：结构相似性指数
- **UIQM**：水下图像质量指标

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 致谢

- 基于 PyTorch 深度学习框架
- 使用 Swin Transformer 作为骨干网络
- 参考了多种水下图像增强方法

## 联系方式

如有问题，请通过 GitHub Issues 联系。