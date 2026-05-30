<img width="1440" height="1748" alt="image" src="https://github.com/user-attachments/assets/0ca1fe86-1c26-4814-a933-6aca47081d3c" /># RadioDiff-k² 📡


<style>
  .rd-root { padding: 1.5rem 0; font-family: var(--font-sans); }
  .rd-header { display: flex; align-items: center; gap: 12px; margin-bottom: 0.25rem; }
  .rd-title { font-size: 22px; font-weight: 500; color: var(--color-text-primary); }
  .rd-sub { font-size: 14px; color: var(--color-text-secondary); margin-bottom: 1.75rem; }
  .rd-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }
  .rd-card { background: var(--color-background-primary); border: 0.5px solid var(--color-border-tertiary); border-radius: var(--border-radius-lg); padding: 1rem 1.1rem 0.9rem; display: flex; flex-direction: column; gap: 6px; }
  .rd-card:hover { border-color: var(--color-border-secondary); }
  .rd-card-head { display: flex; align-items: flex-start; gap: 10px; }
  .rd-icon { width: 36px; height: 36px; border-radius: var(--border-radius-md); display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; margin-top: 1px; }
  .rd-icon.purple { background: #EEEDFE; color: #3C3489; }
  .rd-icon.teal   { background: #E1F5EE; color: #085041; }
  .rd-icon.blue   { background: #E6F1FB; color: #0C447C; }
  .rd-icon.coral  { background: #FAECE7; color: #712B13; }
  .rd-icon.amber  { background: #FAEEDA; color: #633806; }
  .rd-icon.green  { background: #EAF3DE; color: #27500A; }
  .rd-icon.pink   { background: #FBEAF0; color: #72243E; }
  .rd-icon.gray   { background: #F1EFE8; color: #444441; }
  .rd-icon.red    { background: #FCEBEB; color: #791F1F; }
  @media (prefers-color-scheme: dark) {
    .rd-icon.purple { background: #26215C; color: #CECBF6; }
    .rd-icon.teal   { background: #04342C; color: #9FE1CB; }
    .rd-icon.blue   { background: #042C53; color: #B5D4F4; }
    .rd-icon.coral  { background: #4A1B0C; color: #F5C4B3; }
    .rd-icon.amber  { background: #412402; color: #FAC775; }
    .rd-icon.green  { background: #173404; color: #C0DD97; }
    .rd-icon.pink   { background: #4B1528; color: #F4C0D1; }
    .rd-icon.gray   { background: #2C2C2A; color: #D3D1C7; }
    .rd-icon.red    { background: #501313; color: #F7C1C1; }
  }
  .rd-name { font-size: 14px; font-weight: 500; color: var(--color-text-primary); }
  .rd-desc { font-size: 13px; color: var(--color-text-secondary); line-height: 1.5; flex: 1; }
  .rd-footer { display: flex; align-items: center; gap: 8px; margin-top: 4px; flex-wrap: wrap; }
  .rd-badge { font-size: 11px; padding: 2px 8px; border-radius: 99px; font-weight: 500; white-space: nowrap; }
  .badge-tccn  { background: #E1F5EE; color: #0F6E56; }
  .badge-jsac  { background: #E6F1FB; color: #185FA5; }
  .badge-twc   { background: #EEEDFE; color: #534AB7; }
  .badge-tnse  { background: #EAF3DE; color: #3B6D11; }
  .badge-infocom { background: #FAEEDA; color: #854F0B; }
  .badge-icc   { background: #FBEAF0; color: #993556; }
  .badge-arxiv { background: #F1EFE8; color: #5F5E5A; }
  .badge-award { background: #FAECE7; color: #993C1D; }
  @media (prefers-color-scheme: dark) {
    .badge-tccn  { background: #04342C; color: #9FE1CB; }
    .badge-jsac  { background: #042C53; color: #B5D4F4; }
    .badge-twc   { background: #26215C; color: #CECBF6; }
    .badge-tnse  { background: #173404; color: #C0DD97; }
    .badge-infocom { background: #412402; color: #FAC775; }
    .badge-icc   { background: #4B1528; color: #F4C0D1; }
    .badge-arxiv { background: #2C2C2A; color: #D3D1C7; }
    .badge-award { background: #4A1B0C; color: #F5C4B3; }
  }
  .rd-links { display: flex; align-items: center; gap: 6px; margin-left: auto; }
  .rd-link { font-size: 12px; color: var(--color-text-secondary); text-decoration: none; display: flex; align-items: center; gap: 3px; border: 0.5px solid var(--color-border-tertiary); border-radius: var(--border-radius-md); padding: 2px 7px; }
  .rd-link:hover { border-color: var(--color-border-secondary); color: var(--color-text-primary); }
  .rd-link i { font-size: 13px; }
  .rd-awesome { margin-top: 1.5rem; background: var(--color-background-secondary); border-radius: var(--border-radius-lg); padding: 0.85rem 1.1rem; display: flex; align-items: center; gap: 12px; }
  .rd-awesome i { font-size: 20px; color: var(--color-text-secondary); }
  .rd-awesome-text { font-size: 13px; color: var(--color-text-secondary); }
  .rd-awesome-text a { color: var(--color-text-info); text-decoration: none; }
  .rd-awesome-text a:hover { text-decoration: underline; }
</style>

<div class="rd-root">
  <h2 class="sr-only" style="position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0);">RadioDiff family of papers: radio map construction using generative diffusion models</h2>
  <div class="rd-header">
    <i class="ti ti-radio" aria-hidden="true" style="font-size:24px; color:var(--color-text-secondary);"></i>
    <span class="rd-title">RadioDiff family</span>
  </div>
  <p class="rd-sub">Radio map construction via generative diffusion models &nbsp;·&nbsp; UNIC Lab, Xidian University</p>

  <div class="rd-grid">

    <!-- RadioDiff -->
    <div class="rd-card">
      <div class="rd-card-head">
        <div class="rd-icon purple"><i class="ti ti-wave-sine" aria-hidden="true"></i></div>
        <div>
          <p class="rd-name">RadioDiff</p>
          <p class="rd-desc">Base backbone for diffusion-based radio map construction.</p>
        </div>
      </div>
      <div class="rd-footer">
        <span class="rd-badge badge-tccn">IEEE TCCN 2025</span>
        <div class="rd-links">
          <a class="rd-link" href="https://ieeexplore.ieee.org/document/10764739" target="_blank"><i class="ti ti-file-text" aria-hidden="true"></i>Paper</a>
          <a class="rd-link" href="https://github.com/UNIC-Lab/RadioDiff" target="_blank"><i class="ti ti-brand-github" aria-hidden="true"></i>Code</a>
        </div>
      </div>
    </div>

    <!-- RadioDiff-k² -->
    <div class="rd-card">
      <div class="rd-card-head">
        <div class="rd-icon teal"><i class="ti ti-atom" aria-hidden="true"></i></div>
        <div>
          <p class="rd-name">RadioDiff-k²</p>
          <p class="rd-desc">Physics-informed diffusion enhanced by the Helmholtz equation.</p>
        </div>
      </div>
      <div class="rd-footer">
        <span class="rd-badge badge-jsac">IEEE JSAC 2026</span>
        <div class="rd-links">
          <a class="rd-link" href="https://ieeexplore.ieee.org/document/11278649" target="_blank"><i class="ti ti-file-text" aria-hidden="true"></i>Paper</a>
          <a class="rd-link" href="https://github.com/UNIC-Lab/RadioDiff-k" target="_blank"><i class="ti ti-brand-github" aria-hidden="true"></i>Code</a>
        </div>
      </div>
    </div>

    <!-- RadioDiff-Turbo -->
    <div class="rd-card">
      <div class="rd-card-head">
        <div class="rd-icon amber"><i class="ti ti-bolt" aria-hidden="true"></i></div>
        <div>
          <p class="rd-name">RadioDiff-Turbo</p>
          <p class="rd-desc">Efficiency-enhanced RadioDiff for accelerated inference.</p>
        </div>
      </div>
      <div class="rd-footer">
        <span class="rd-badge badge-infocom">IEEE INFOCOM Wksp 2025</span>
        <div class="rd-links">
          <a class="rd-link" href="https://ieeexplore.ieee.org/abstract/document/11152929/" target="_blank"><i class="ti ti-file-text" aria-hidden="true"></i>Paper</a>
        </div>
      </div>
    </div>

    <!-- RadioDiff-Flux -->
    <div class="rd-card">
      <div class="rd-card-head">
        <div class="rd-icon blue"><i class="ti ti-refresh" aria-hidden="true"></i></div>
        <div>
          <p class="rd-name">RadioDiff-Flux</p>
          <p class="rd-desc">Handles dynamic environments and base station location changes.</p>
        </div>
      </div>
      <div class="rd-footer">
        <span class="rd-badge badge-tccn">IEEE TCCN 2026</span>
        <div class="rd-links">
          <a class="rd-link" href="https://ieeexplore.ieee.org/document/11282987/" target="_blank"><i class="ti ti-file-text" aria-hidden="true"></i>Paper</a>
        </div>
      </div>
    </div>

    <!-- RadioDiff-FS -->
    <div class="rd-card">
      <div class="rd-card-head">
        <div class="rd-icon green"><i class="ti ti-sparkles" aria-hidden="true"></i></div>
        <div>
          <p class="rd-name">RadioDiff-FS</p>
          <p class="rd-desc">Few-shot learning for radio map construction with limited measurements.</p>
        </div>
      </div>
      <div class="rd-footer">
        <span class="rd-badge badge-arxiv">arXiv preprint</span>
        <div class="rd-links">
          <a class="rd-link" href="https://arxiv.org/abs/2603.18865" target="_blank"><i class="ti ti-file-text" aria-hidden="true"></i>Paper</a>
          <a class="rd-link" href="https://github.com/UNIC-Lab/RadioDiff-FS" target="_blank"><i class="ti ti-brand-github" aria-hidden="true"></i>Code</a>
        </div>
      </div>
    </div>

    <!-- iRadioDiff -->
    <div class="rd-card">
      <div class="rd-card-head">
        <div class="rd-icon pink"><i class="ti ti-building" aria-hidden="true"></i></div>
        <div>
          <p class="rd-name">iRadioDiff</p>
          <p class="rd-desc">Indoor radio map construction with physical information integration.</p>
        </div>
      </div>
      <div class="rd-footer">
        <span class="rd-badge badge-icc">IEEE ICC 2026</span>
        <span class="rd-badge badge-award">🏆 Best Paper</span>
        <div class="rd-links">
          <a class="rd-link" href="https://arxiv.org/abs/2511.20015" target="_blank"><i class="ti ti-file-text" aria-hidden="true"></i>Paper</a>
          <a class="rd-link" href="https://github.com/UNIC-Lab/iRadioDiff" target="_blank"><i class="ti ti-brand-github" aria-hidden="true"></i>Code</a>
        </div>
      </div>
    </div>

    <!-- RadioDiff-3D -->
    <div class="rd-card">
      <div class="rd-card-head">
        <div class="rd-icon coral"><i class="ti ti-3d-cube-sphere" aria-hidden="true"></i></div>
        <div>
          <p class="rd-name">RadioDiff-3D</p>
          <p class="rd-desc">3D radio map construction with the UrbanRadio3D dataset.</p>
        </div>
      </div>
      <div class="rd-footer">
        <span class="rd-badge badge-tnse">IEEE TNSE 2025</span>
        <div class="rd-links">
          <a class="rd-link" href="https://ieeexplore.ieee.org/document/11083758" target="_blank"><i class="ti ti-file-text" aria-hidden="true"></i>Paper</a>
          <a class="rd-link" href="https://github.com/UNIC-Lab/UrbanRadio3D" target="_blank"><i class="ti ti-brand-github" aria-hidden="true"></i>Code</a>
        </div>
      </div>
    </div>

    <!-- RadioDiff-Inverse -->
    <div class="rd-card">
      <div class="rd-card-head">
        <div class="rd-icon gray"><i class="ti ti-antenna" aria-hidden="true"></i></div>
        <div>
          <p class="rd-name">RadioDiff-Inverse</p>
          <p class="rd-desc">Sparse measurement-based radio map recovery for ISAC applications.</p>
        </div>
      </div>
      <div class="rd-footer">
        <span class="rd-badge badge-twc">IEEE TWC 2026</span>
        <div class="rd-links">
          <a class="rd-link" href="https://arxiv.org/abs/2504.14298" target="_blank"><i class="ti ti-file-text" aria-hidden="true"></i>Paper</a>
          <a class="rd-link" href="https://github.com/UNIC-Lab/radiodiff-inverse" target="_blank"><i class="ti ti-brand-github" aria-hidden="true"></i>Code</a>
        </div>
      </div>
    </div>

    <!-- RadioDiff-Loc -->
    <div class="rd-card">
      <div class="rd-card-head">
        <div class="rd-icon red"><i class="ti ti-map-pin" aria-hidden="true"></i></div>
        <div>
          <p class="rd-name">RadioDiff-Loc</p>
          <p class="rd-desc">Sparse measurement-based NLoS localization using diffusion models.</p>
        </div>
      </div>
      <div class="rd-footer">
        <span class="rd-badge badge-arxiv">arXiv preprint</span>
        <div class="rd-links">
          <a class="rd-link" href="https://www.arxiv.org/abs/2509.01875" target="_blank"><i class="ti ti-file-text" aria-hidden="true"></i>Paper</a>
        </div>
      </div>
    </div>

  </div>

  <div class="rd-awesome">
    <i class="ti ti-stars" aria-hidden="true"></i>
    <p class="rd-awesome-text">
      For a comprehensive categorized overview of radio map research, visit
      <a href="https://github.com/UNIC-Lab/Awesome-Radio-Map-Categorized" target="_blank">Awesome-Radio-Map-Categorized <i class="ti ti-external-link" style="font-size:12px;vertical-align:-1px;" aria-hidden="true"></i></a>
    </p>
  </div>
</div>


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
