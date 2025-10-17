# RadioDiff-k2 📡
---
### Welcome to the RadioDiff family

Base BackBone, Paper Link: [RadioDiff](https://ieeexplore.ieee.org/document/10764739), Code Link: [GitHub](https://github.com/UNIC-Lab/RadioDiff)

PINN Enhanced with Helmholtz Equation, Paper Link: [RadioDiff-$k^2$](https://arxiv.org/pdf/2504.15623), Code Link: [GitHub](https://github.com/UNIC-Lab/RadioDiff-k)

Efficiency Enhanced RadioDiff, Paper Link: [RadioDiff-Turbo](https://ieeexplore.ieee.org/abstract/document/11152929/)

Indoor RM Construction with Physical Information, Code Link: [GitHub](https://github.com/UNIC-Lab/iRadioDiff)

3D RM with DataSet, Paper Link: [RadioDiff-3D](https://ieeexplore.ieee.org/document/11083758), Code Link: [GitHub](https://github.com/UNIC-Lab/UrbanRadio3D)

Sparse Measurement for RM ISAC, Paper Link: [RadioDiff-Inverse](https://arxiv.org/abs/2504.14298)

Sparse Measurement for NLoS Localization, Paper Link: [RadioDiff-Loc](https://www.arxiv.org/abs/2509.01875)

For more RM information, please visit the repo of [Awesome-Radio-Map-Categorized](https://github.com/UNIC-Lab/Awesome-Radio-Map-Categorized)

---


> 基于扩散模型的智能无线电地图重建系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

RadioDiff-k2 是一个先进的无线电地图重建项目，使用条件扩散模型从稀疏测量数据中生成高质量的无线电覆盖地图。该项目特别适用于5G/6G网络规划、信号传播预测和网络优化。

## ✨ 主要特性

### 🎯 多种仿真方法
- **DPM** - 确定性传播模型，快速准确
- **IRT4** - 迭代射线追踪，高精度预测  
- **DPMCARK** - 结合车辆信息的增强模型

### 🏗️ 先进架构
- **条件扩散模型** - 基于Swin Transformer
- **VAE编码器** - 高效特征压缩
- **多尺度处理** - 支持不同分辨率

### 📊 丰富特征
- **建筑物布局** - 城市环境建模
- **发射器位置** - 信号源信息
- **车辆数据** - 动态障碍物
- **K2特征** - 物理传播特性

## 🚀 快速开始

### 环境要求
```bash
Python >= 3.8
CUDA >= 11.0
PyTorch >= 1.12
```



## 📁 项目结构

```
RadioDiff-k2/
├── 📋 configs/                    # 配置文件
│   ├── BSDS_sample_*.yaml         # 推理配置
│   └── BSDS_train_*.yaml          # 训练配置
├── 🧠 denoising_diffusion_pytorch/ # 扩散模型核心
├── 🔧 lib/                        # 工具库
│   ├── loaders.py                 # 数据加载器
│   └── modules.py                 # 网络模块
├── 💾 model/                      # 预训练模型
├── 📊 inference/                  # 推理结果
│   ├── DPMCARK/                  # DPMCARK方法结果
│   ├── DPMK/                     # DPMK方法结果
│   └── IRT4K/                    # IRT4K方法结果
├── 📈 metrics/                    # 评估指标
├── ⚡ TFMQ/                      # 量化优化
├── 🚀 train_cond_ldm.py          # 训练脚本
├── 🔮 sample_cond_ldm.py         # 推理脚本
├── 🏗️ train_vae.py              # VAE训练
├── 🧮 caculate_k.py              # K2特征计算
├── 🎯 demo.py                    # 使用示例
├── 📦 requirements.txt           # 依赖列表
└── 📖 README.md                  # 项目文档
```

<!-- ## 📦 安装指南

### 方法一：pip安装（推荐）
```bash
# 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install accelerate torchmetrics scikit-image opencv-python
pip install pyyaml tqdm matplotlib pandas pillow

# 安装可选依赖
pip install tensorboard wandb  # 用于训练监控
``` -->

### 方法二：conda安装
<!-- ```bash
# 创建环境
conda create -n radiodiff python=3.9
conda activate radiodiff

# 安装PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install accelerate torchmetrics scikit-image opencv-python pyyaml tqdm matplotlib pandas pillow
``` -->

## 🎯 使用指南

### 1️⃣ 数据准备

#### 数据集结构
```
RadioMapSeer/
├── 📁 png/
│   ├── buildings_complete/        # 建筑物图像 (256x256)
│   ├── antennas/                  # 发射器位置 (256x256)
│   └── cars/                      # 车辆信息 (可选)
├── 📁 gain/
│   ├── DPM/                       # DPM仿真结果
│   ├── IRT4/                      # IRT4仿真结果
│   └── IRT4_k2_neg_norm/          # K2特征图
└── 📁 metadata/                   # 元数据文件
```

#### 生成K2特征
```bash
# 运行K2特征计算脚本
python caculate_k.py
```

### 2️⃣ 模型训练



#### 步骤1：训练条件扩散模型
```bash
# 训练主模型
python train_cond_ldm.py --cfg configs/BSDS_train_DPMK.yaml

python train_cond_ldm.py --cfg configs/BSDS_train_DPMCARK.yaml

python train_cond_ldm.py --cfg configs/BSDS_train_IRT4K.yaml

```

### 3️⃣ 模型推理

#### 基础推理
```bash
# 使用DPMCARK方法推理
python sample_cond_ldm.py --cfg configs/BSDS_sample_DPMCARK.yaml

# 使用DPMK方法推理
python sample_cond_ldm.py --cfg configs/BSDS_sample_DPMK.yaml

# 使用IRT4K方法推理
python sample_cond_ldm.py --cfg configs/BSDS_sample_IRT4K.yaml
```

