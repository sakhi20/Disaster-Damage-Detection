# Disaster Damage Detection via Satellite Imagery

> A two-phase deep learning pipeline for automated building damage assessment from satellite imagery, achieving an overall F1 of **0.8136** on the xBD benchmark — surpassing the xView2 Challenge winner with a single model.

**Mulya S. Patel · Vivek K. Vanera · Sakhi T. Patel**  
*North Carolina State University · Advanced Topics in Machine Learning · Spring 2026*

![F1 Score](https://img.shields.io/badge/Overall_F1-0.8136-brightgreen) ![Loc F1](https://img.shields.io/badge/Localization_F1-0.8752-brightgreen) ![Dataset](https://img.shields.io/badge/Dataset-xBD-blue) ![Framework](https://img.shields.io/badge/Framework-PyTorch-orange) ![Institution](https://img.shields.io/badge/Institution-NC_State-red)

---

## Overview

After a disaster, emergency responders need to know which buildings are damaged — within hours, not weeks. Satellite imagery is available almost immediately after any event worldwide, but manually inspecting thousands of images at scale is infeasible.

We address this with a **two-phase pipeline**:

1. **Phase 1 — Building Localization:** Fine-tune a geospatial foundation model on pre-disaster satellite images to produce per-pixel building footprint masks.
2. **Phase 2 — Damage Classification:** Extend the Phase 1 encoder into a dual-branch architecture (`JointDamageNet`) that takes pre + post-disaster image pairs and jointly predicts building location and five-class damage severity.

---

## Results

### Phase 2 — Damage Classification (xBD hold split)

| Method | Overall F1 | Damage F1 | Loc F1 |
|--------|:---------:|:---------:|:------:|
| ChangeMamba — SOTA (IEEE TGRS 2024) | 0.8141 | 0.7884 | 0.8738 |
| xView2 First-Place (42-model ensemble) | 0.8112 | 0.7887 | 0.8635 |
| ChangeOS | 0.7857 | 0.7564 | 0.8541 |
| DamFormer | 0.7702 | 0.7281 | 0.7281 |
| **Ours — JointDamageNet (single model)†** | **0.8136** | **0.7873** | **0.8752** |

*† Single model. All other methods use ensembles or are evaluated under identical hold-split conditions.*

Our single model achieves the **highest localization F1 (0.8752)** of all compared methods and a **minor-damage F1 of 0.589** — more than double DisasterAdaptiveNet's 0.253 on the same evaluation split.

### Phase 1 — Foundation Model Comparison (xBD hold split)

| Model | Pretrained On | Val Dice | Val IoU |
|-------|--------------|:--------:|:-------:|
| **DeepLabV3+ (selected)** | **ImageNet** | **0.877** | **0.781** |
| DINOv2 | LVD-142M (natural images) | 0.874 | 0.776 |
| DeepLabV3+ | fMoW (geospatial) | 0.873 | 0.775 |
| SAM2 | SA-1B (GT prompts†) | 0.869 | 0.768 |
| CrossEarth | Multi-domain remote sensing | 0.861 | 0.755 |
| Satlas | Satlas-pretrain (aerial) | 0.819 | 0.747 |
| SegFormer | ADE20K | 0.628 | — |

*† Upper bound — uses ground-truth building polygon centroids as prompts; not practical at deployment.*

**Key finding:** An ImageNet-pretrained backbone outperformed geospatial-specific models (fMoW, Satlas). Models pretrained for scene-level classification do not automatically transfer to dense pixel-level segmentation.

---

## Architecture

### Phase 1 — DeepLabV3+
Standard DeepLabV3+ with a **ResNet-50 backbone** (ImageNet pretrained), trained on pre-disaster images for binary building segmentation.
- Loss: BCE with Logits + Dice
- Optimizer: AdamW + CosineAnnealingLR
- Input: 512×512 tiles (xBD images are 1024×1024, solved with tiling)

### Phase 2 — JointDamageNet

A **shared ResNet-50 encoder** (weights initialized from Phase 1) processes both pre- and post-disaster images and feeds two asymmetric decoder heads:

```
Pre-disaster image  ──┐
                      ├──► Shared ResNet-50 Encoder ──► Low-level (H/4)  ──┐
Post-disaster image ──┘                               └──► High-level (H/16) ──┤
                                                                              │
           Localization Head: cat(pre, post) ──────────────────────────────► Binary mask
           Damage Head: cat(pre, post, post−pre) ──► learned compression ──► 5-class map
```

**Three design contributions over baseline:**
1. **Asymmetric fusion** — localization uses `cat(pre, post)`; damage additionally receives the explicit change signal `post − pre`, compressed back to 2× width via a learned 1×1 conv.
2. **Ordinal loss (EMD)** — Earth Mover's Distance penalizes predictions proportional to their distance from the true class on the ordinal damage scale.
3. **Two-level imbalance correction** — `WeightedRandomSampler` (minority-damage tiles upsampled 3×) combined with per-pixel inverse-frequency class weights in the Focal Loss.

---

## 🔑 Key Technical Contributions

1. **Phase-transfer learning** — encoder weights from Phase 1 building localization directly initialize Phase 2 damage classification, enabling faster convergence and better feature reuse.
2. **Asymmetric dual-branch fusion** — localization head uses `cat(pre, post)`; damage head additionally receives the explicit change signal `post − pre` compressed via a learned 1×1 conv, letting each task access the information it needs.
3. **Ordinal EMD loss** — Earth Mover's Distance penalizes damage misclassifications proportionally to their ordinal distance, reflecting the real cost of confusing "No Damage" with "Destroyed."
4. **Two-level class imbalance handling** — `WeightedRandomSampler` upsamples minority-damage tiles 3× at the batch level; per-pixel inverse-frequency weights in Focal Loss address imbalance within each tile.
5. **Systematic ablation** — five experiments (expA→expE) isolate the contribution of each design choice, following standard ML research methodology.

---

## Dataset

[xBD](https://xview2.org/dataset) — released by DIUx for the xView2 Challenge

| Stat | Value |
|------|-------|
| Building polygons | 850,000+ |
| Disaster events | 19 (wildfires, hurricanes, earthquakes, floods, tsunamis, volcanic eruptions) |
| Image resolution | 0.5 m/pixel RGB, 1024×1024 px |
| Image pairs | Co-registered pre- and post-disaster |
| Damage labels | 0 = Background · 1 = No Damage · 2 = Minor · 3 = Major · 4 = Destroyed |

---

## Project Structure

```
├── README.md
├── requirements.txt
├── .gitignore
│
├── phase1/                        # Building localization
│   ├── model.py                   # DeepLabV3+ — ImageNet backbone
│   ├── model_fmow.py              # DeepLabV3+ — fMoW geospatial backbone
│   ├── dataset.py                 # XBDSegDataset — 512x512 tiling with building-density filter
│   ├── train.py                   # BCE + Dice loss, AdamW, CosineAnnealingLR
│   ├── evaluate.py                # IoU / Dice evaluation
│   └── visualize.py               # Prediction overlay visualization
│
├── phase2/                        # Damage classification
│   ├── losses.py                  # BinaryFocalLoss, FocalLoss, EMDLoss (ordinal)
│   ├── expA/                      # Baseline — joint model, cat(pre,post) fusion
│   ├── expB/                      # + AMP, differential LRs, warmup+cosine schedule
│   ├── expC/                      # + Pixel-level supervised contrastive loss
│   ├── expD/                      # + Asymmetric diff-fusion, EMD loss, weighted sampler  <- best
│   └── expE/                      # + Minor-class boost, label smoothing, early stopping
│
└── notebooks/
    ├── phase1_experiments.ipynb   # Phase 1 exploration and ablations
    └── phase2_experiments.ipynb   # Phase 2 exploration and ablations
```

---

## Setup & Usage

```bash
git clone https://github.com/sakhi20/Disaster-Damage-Detection.git
cd Disaster-Damage-Detection
pip install -r requirements.txt
```

Download and extract the [xBD dataset](https://xview2.org/dataset) into an `xbd/` directory at the project root:

```
xbd/
├── tier1/  ├── tier3/
└── hold/
    └── (each split has images/ and masks/ subdirectories)
```

**Train Phase 1 (building localization):**
```bash
cd main-codebase/phase1/deeplabsv3+
python train.py
# Saves best_model.pth — encoder weights are transferred to Phase 2
```

**Train Phase 2 (damage classification — Experiment D, best):**
```bash
cd main-codebase/phase2/expD
python train.py    # Requires best_model.pth from Phase 1 in the working directory
```

---

## Reproducibility Studies

Before building our own system, we reproduced two competitive baselines to calibrate our evaluation:

| Method | Split | Our Score | Reported |
|--------|-------|:---------:|:--------:|
| xView2 First-Place | xBD random | 0.788 | 0.811 |
| DisasterAdaptiveNet | xBD event-based | 0.533 | 0.541 |
| Strong Baseline | xBD event-based | 0.478 | 0.483 |

The xView2 gap (0.023) is explained by training 12 backbone models instead of 42 due to compute constraints — not a fundamental reproducibility failure. DisasterAdaptiveNet gaps are attributable to minor preprocessing differences.

---

## Team

| Name | Email |
|------|-------|
| Mulya S. Patel | mspate22@ncsu.edu |
| Vivek K. Vanera | vvanera@ncsu.edu |
| Sakhi T. Patel | stpatel14@ncsu.edu |

---

## 📄 Citation

If you use this work, please cite:
```bibtex
@misc{patel2026disasterdamage,
  title  = {JointDamageNet: A Two-Phase Pipeline for Satellite Imagery
             Disaster Damage Detection},
  author = {Patel, Mulya S. and Vanera, Vivek K. and Patel, Sakhi T.},
  year   = {2026},
  note   = {NC State University, Advanced Topics in Machine Learning}
}
```
> **Note:** An arXiv preprint is in preparation.

---

## References

1. V. Durnov, "xView2 first place solution," 2019. [[GitHub]](https://github.com/DIUx-xView/xView2_first_place)
2. S. Hafner et al., "DisasterAdaptiveNet: A robust network for multi-hazard building damage detection," *Int. J. Applied Earth Observation*, 2025.
3. L.-C. Chen et al., "Encoder-decoder with atrous separable convolution for semantic image segmentation (DeepLabV3+)," *ECCV*, 2018.
4. K. Chen et al., "ChangeMamba: Remote sensing change detection with spatiotemporal state space model," *IEEE TGRS*, 2024.
5. M. Oquab et al., "DINOv2: Learning robust visual features without supervision," *TMLR*, 2024.
