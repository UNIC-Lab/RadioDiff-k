## 📡 Welcome to the RadioDiff Family

> Radio map construction via generative diffusion models — UNIC Lab, Xidian University

---

### 🔷 Base Backbone

**RadioDiff** — *The foundational diffusion model for radio map construction.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/document/10764739) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/RadioDiff) &nbsp;|&nbsp; ![IEEE TCCN](https://img.shields.io/badge/IEEE-TCCN%202025-blue)

---

### 🔬 Physics-Informed Extensions

**RadioDiff-k²** — *PINN-enhanced diffusion guided by the Helmholtz equation.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/document/11278649) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/RadioDiff-k) &nbsp;|&nbsp; ![IEEE JSAC](https://img.shields.io/badge/IEEE-JSAC%202026-blue)

**iRadioDiff** — *Indoor radio map construction with physical information integration.*
&nbsp;&nbsp;📄 [Paper](https://arxiv.org/abs/2511.20015) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/iRadioDiff) &nbsp;|&nbsp; ![IEEE ICC](https://img.shields.io/badge/IEEE-ICC%202026-blue) &nbsp;![Best Paper](https://img.shields.io/badge/🏆-Best%20Paper%20Award-orange)

---

### ⚡ Efficiency & Dynamics

**RadioDiff-Turbo** — *Efficiency-enhanced RadioDiff for accelerated inference.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/abstract/document/11152929/) &nbsp;|&nbsp; ![INFOCOM Workshop](https://img.shields.io/badge/IEEE-INFOCOM%20Wksp%202025-lightgrey)

**RadioDiff-Flux** — *Adaptive reconstruction under dynamic environments and base station location changes.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/document/11282987/) &nbsp;|&nbsp; ![IEEE TCCN](https://img.shields.io/badge/IEEE-TCCN%202026-blue)

---

### 🌐 Extended Scenarios

**RadioDiff-3D** — *3D radio map construction with the UrbanRadio3D dataset.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/document/11083758) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/UrbanRadio3D) &nbsp;|&nbsp; ![IEEE TNSE](https://img.shields.io/badge/IEEE-TNSE%202025-blue)

**RadioDiff-FS** — *Few-shot learning for radio map construction with limited measurements.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/document/11577136) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/RadioDiff-FS) &nbsp;|&nbsp; ![IEEE IoTJ](https://img.shields.io/badge/IEEE-IoTJ%202026-blue)

---

### 📶 Sparse Measurement & Localization

**RadioDiff-Inverse** — *Sparse measurement-based radio map recovery for ISAC applications.*
&nbsp;&nbsp;📄 [Paper](https://arxiv.org/abs/2504.14298) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/radiodiff-inverse) &nbsp;|&nbsp; ![IEEE TWC](https://img.shields.io/badge/IEEE-TWC%202026-blue)

**RadioDiff-Loc** — *Sparse measurement-based NLoS localization using diffusion models.*
&nbsp;&nbsp;📄 [Paper](https://www.arxiv.org/abs/2509.01875) &nbsp;|&nbsp; ![arXiv](https://img.shields.io/badge/arXiv-preprint-lightgrey)

---

> 📚 For a comprehensive categorized overview of radio map research, visit [**Awesome-Radio-Map-Categorized**](https://github.com/UNIC-Lab/Awesome-Radio-Map-Categorized).


### 🎉🎉🎉 The paper has been accepted by IEEE JSAC!

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
