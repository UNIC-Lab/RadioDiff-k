# RadioDiff-k² 📡

---
### Welcome to the RadioDiff family

Base BackBone, Paper Link: [RadioDiff](https://ieeexplore.ieee.org/document/10764739), Code Link: [GitHub](https://github.com/UNIC-Lab/RadioDiff)

PINN Enhanced with Helmholtz Equation, Paper Link: [RadioDiff-$k^2$](https://arxiv.org/pdf/2504.15623), Code Link: [GitHub](https://github.com/UNIC-Lab/RadioDiff-k)

Efficiency Enhanced RadioDiff, Paper Link: [RadioDiff-Turbo](https://ieeexplore.ieee.org/abstract/document/11152929/)

Indoor RM Construction with Physical Information, Paper Link: [iRadioDiff](https://arxiv.org/abs/2511.20015), Code Link: [GitHub](https://github.com/UNIC-Lab/iRadioDiff)

3D RM with DataSet, Paper Link: [RadioDiff-3D](https://ieeexplore.ieee.org/document/11083758), Code Link: [GitHub](https://github.com/UNIC-Lab/UrbanRadio3D)

Sparse Measurement for RM ISAC, Paper Link: [RadioDiff-Inverse](https://arxiv.org/abs/2504.14298)

Sparse Measurement for NLoS Localization, Paper Link: [RadioDiff-Loc](https://www.arxiv.org/abs/2509.01875)

For more RM information, please visit the repo of [Awesome-Radio-Map-Categorized](https://github.com/UNIC-Lab/Awesome-Radio-Map-Categorized)

---

> An intelligent radio-map reconstruction system based on diffusion models. 📶✨ 

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

RadioDiff-k² is an advanced radio-map reconstruction project that leverages conditional diffusion models to generate high-quality radio-coverage maps from sparse measurements. The project serves 5G and 6G network planning, propagation prediction, and network optimization. 🚀📡 

## ✨ Key Features

### 🎯 Multiple Simulation Methods

* **DPM** — deterministic propagation modeling with high speed and accuracy
* **IRT4** — iterative ray tracing with high-precision prediction
* **DPMCARK** — vehicle-aware enhancement for urban mobility scenes 🚗📡 

### 🏗️ Advanced Architecture

* **Conditional diffusion model** built on Swin Transformer
* **VAE encoder** for compact and efficient representation
* **Multi-scale processing** for flexible resolution support 🧠🧩 

### 📊 Rich Conditioning Features

* **Building layouts** for realistic urban environments
* **Transmitter positions** to capture source attributes
* **Vehicle data** for dynamic occlusions
* **k² features** to encode physical propagation traits 🏙️📍🚘📐 

## 🚀 Quick Start

### Environment Requirements

```bash
Python >= 3.8
CUDA >= 11.0
PyTorch >= 1.12
```



## 📁 Project Structure

```
RadioDiff-k2/
├── 📋 configs/                    # Configuration files
│   ├── BSDS_sample_*.yaml         # Inference configs
│   └── BSDS_train_*.yaml          # Training configs
├── 🧠 denoising_diffusion_pytorch/ # Diffusion core
├── 🔧 lib/                        # Utilities
│   ├── loaders.py                 # Data loaders
│   └── modules.py                 # Network modules
├── 💾 model/                      # Pretrained models
├── 📊 inference/                  # Inference results
│   ├── DPMCARK/                   # DPMCARK outputs
│   ├── DPMK/                      # DPMK outputs
│   └── IRT4K/                     # IRT4K outputs
├── 📈 metrics/                    # Evaluation metrics
├── 🚀 train_cond_ldm.py           # Training script
├── 🔮 sample_cond_ldm.py          # Inference script
├── 🏗️ train_vae.py               # VAE training
├── 🧮 caculate_k.py               # k² feature computation
├── 🎯 demo.py                     # Usage examples
├── 📦 requirements.txt            # Dependencies
└── 📖 README.md                   # Project docs
```



<!--
## 📦 Installation Guide

### Method 1: pip
```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install accelerate torchmetrics scikit-image opencv-python
pip install pyyaml tqdm matplotlib pandas pillow

# Optional tools for monitoring
pip install tensorboard wandb
```
-->

### Method 2: conda

<!--
```bash
# Create environment
conda create -n radiodiff python=3.9
conda activate radiodiff

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Other dependencies
pip install accelerate torchmetrics scikit-image opencv-python pyyaml tqdm matplotlib pandas pillow
```
-->



## 🎯 Usage Guide

### 1️⃣ Data Preparation

#### Dataset Layout

```
RadioMapSeer/
├── 📁 png/
│   ├── buildings_complete/        # Building images 256x256
│   ├── antennas/                  # Transmitter positions 256x256
│   └── cars/                      # Vehicle information optional
├── 📁 gain/
│   ├── DPM/                       # DPM simulation results
│   ├── IRT4/                      # IRT4 simulation results
│   └── IRT4_k2_neg_norm/          # k² feature maps
└── 📁 metadata/                   # Meta files
```



#### Generate k² Features

```bash
# Run the k² feature computation script
python caculate_k.py
```



### 2️⃣ Model Training

#### Step 1 — Train the conditional diffusion model

```bash
# Train the main model
python train_cond_ldm.py --cfg configs/BSDS_train_DPMK.yaml
python train_cond_ldm.py --cfg configs/BSDS_train_DPMCARK.yaml
python train_cond_ldm.py --cfg configs/BSDS_train_IRT4K.yaml
```



### 3️⃣ Inference

#### Basic Inference

```bash
# DPMCARK inference
python sample_cond_ldm.py --cfg configs/BSDS_sample_DPMCARK.yaml

# DPMK inference
python sample_cond_ldm.py --cfg configs/BSDS_sample_DPMK.yaml

# IRT4K inference
python sample_cond_ldm.py --cfg configs/BSDS_sample_IRT4K.yaml
```
