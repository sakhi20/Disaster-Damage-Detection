# Label-Efficient Disaster Response Using Foundation Models

**Authors:** Sakhi Patel · Vivek Vanera · Mulya Patel  
**Date:** February 19, 2026

## 🚨 The Problem: Disasters Strike, But Labels Take Days

Disaster damage assessment relies on manually labeled satellite/drone imagery, which is slow, expensive, and requires domain experts.

*   **Labeling thousands of images takes days to weeks**
*   The process is time-consuming and mentally exhausting
*   Not scalable during emergency situations
*   Delays critical relief planning and resource allocation

**Goal:** Develop methods that learn effectively from minimal labeled data, enabling faster, scalable, and reliable disaster response.

---

## 🚀 Our Approach: Two-Phase Pipeline

### Phase 1: Foundational Feature Learning (Transfer Learning)
*   **Model:** NASA's **Prithvi-100M** (Geospatial Foundation Model)
*   **Method:** Fine-tune on bi-temporal satellite imagery to classify structural components (Roof, Building, etc.).
*   **Objective:** Learn strong spatial and structural representations with minimal labeled data.

### Phase 2: Damage Detection via Comparative Analysis
*   **Model:** Vision Transformer (ViT) architecture.
*   **Input:** Before and After disaster satellite images.
*   **Method:** Initialize with weights learned in Phase 1 and perform comparative analysis to detect structural damage.
*   **Constraint:** Use only **10% labels** to achieve superior results.

---

## 📊 Current Progress

We have implemented and benchmarked leading change detection architectures:
1.  **Siamese U-Net:** Dual-branch encoder-decoder for pixel-level change mapping.
2.  **SpaU-Net:** Specialized attention mechanisms to focus on critical damage regions.

### Preliminary Results (xBD Dataset)
*   Implemented both architectures.
*   Evaluated under label-constrained settings.
*   Current focus: Hyperparameter optimization and full dataset training.

---

## 📡 Future Directions: SAR Integration

**Why SAR?** Synthetic Aperture Radar (SAR) penetrates clouds and smoke, providing all-weather imaging.

**Integration Plan:**
*   Combine SAR + Optical imagery for richer feature representation.
*   Feed SAR data into Phase 1 transfer-learning model.
*   Enhance Phase 2 Comparative Analysis for better robustness under challenging conditions.

---

## 📚 References
*   Wang, Q., et al. (2023). *High-Resolution Remote Sensing Image Change Detection Method Based on Improved Siamese U-Net*. Remote Sensing.
*   Yu, et al. (2024). *Benchmarking Attention Mechanisms and Consistency Regularization for Post-Flood Building Damage Assessment*. arXiv.
*   NASA Prithvi Model: [huggingface.co/ibm-nasa-geospatial/Prithvi-100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M)
