# Disaster Response AI: Building Damage Detection with Prithvi & Vision Transformers

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project develops a post-disaster building damage detection system using the **xBD dataset** and state-of-the-art geospatial foundation models. We leverage **NASA's Prithvi foundation model** (a geospatial Vision Transformer pre-trained on Harmonized Landsat Sentinel-2 imagery) alongside fine-tuned Vision Transformer (ViT) architectures to classify building damage from satellite imagery captured before and after natural disasters.

### Key Contributions
- Fine-tuning Prithvi (geospatial foundation model) on xBD building damage classification
- Comparative study: Prithvi vs. standard ViT vs. CNN baselines
- Bi-temporal change detection using pre/post-disaster image pairs
- Evaluation on diverse disaster types (hurricanes, wildfires, floods, earthquakes)

---

## Dataset: xBD

The [xBD dataset](https://xview2.org/) contains:
- **~850,000** building polygons across **19 disaster events**
- Satellite imagery at ~0.3m/pixel resolution
- 4-class damage labels: `no-damage`, `minor-damage`, `major-damage`, `destroyed`
- Pre/post-disaster image pairs in GeoTIFF format

---

## Project Structure

```
project/
├── data/                   # xBD dataset (gitignored)
├── models/                 # Model implementations
│   ├── prithvi/            # Prithvi foundation model wrappers
│   ├── vit/                # Vision Transformer implementations
│   └── baselines/          # CNN baseline models
├── experiments/            # Experiment runner scripts
├── utils/                  # Shared utilities
│   ├── data_loader.py      # xBD dataset loading
│   ├── metrics.py          # Evaluation metrics
│   └── visualization.py    # Result visualization
├── configs/                # YAML configuration files
├── notebooks/              # Jupyter exploration notebooks
├── scripts/                # Training & evaluation scripts
├── results/                # Outputs (gitignored)
└── paper/                  # LaTeX paper files
```

---

## Setup

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (for GPU training)
- Conda (recommended)

### Option 1: Conda Environment (Recommended)

```bash
conda env create -f environment.yml
conda activate disaster-response-ai
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

### Download Prithvi Weights

```bash
# Download from HuggingFace
python scripts/download_prithvi.py
```

---

## Quick Start

### Training

```bash
# Train Prithvi fine-tuned model
python scripts/train.py --config configs/prithvi_finetune.yaml

# Train ViT baseline
python scripts/train.py --config configs/vit_baseline.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --config configs/prithvi_finetune.yaml --checkpoint checkpoints/best_model.pth
```

### Experiment Tracking

We use [Weights & Biases](https://wandb.ai/) for experiment tracking:

```bash
wandb login
python scripts/train.py --config configs/prithvi_finetune.yaml --wandb
```

---

## Models

| Model | Backbone | Pre-training | Params |
|-------|----------|-------------|--------|
| Prithvi-100M | ViT-L | HLS Satellite (NASA) | 100M |
| ViT-B/16 | ViT-B | ImageNet-21k | 86M |
| ResNet-50 | CNN | ImageNet | 25M |
| Swin-T | Swin | ImageNet-22k | 28M |

---

## Team

- Sakhi Patel
- Vivek Vanera
- Mulya Patel

---

## References

- Prithvi: [HuggingFace Model Card](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M)
- xBD Dataset: [xView2 Challenge](https://xview2.org/)
- xBD Paper: Gupta et al., "Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery" (CVPR 2019)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
