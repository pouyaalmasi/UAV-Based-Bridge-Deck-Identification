# UAV-Based Bridge Deck Identification using DeepLab v3+

A deep learning-based semantic segmentation framework for automated bridge deck identification from UAV-acquired images using a DeepLab v3+ architecture.

This repository contains:
- MATLAB implementation of the bridge deck segmentation framework
- Pretrained DeepLab v3+ model
- Inference script for testing on new UAV images
- Example visualization pipeline for predicted bridge deck masks

---

# Overview

This repository presents the bridge deck identification component of the paper:

**End-to-End UAV-Enabled Bridge Deck Inspection: From Localization to High-Precision Crack Detection and Quantification**

The objective of this repository is to automatically identify and segment bridge deck regions from UAV-acquired inspection images. Accurate deck localization provides the foundation for downstream damage detection, crack segmentation, and quantitative condition assessment.

The model uses DeepLab v3+ with a ResNet-18 backbone for pixel-level semantic segmentation of bridge decks.

---

# Features

## Bridge Deck Segmentation
- DeepLab v3+ semantic segmentation
- ResNet-18 backbone
- Pixel-level deck localization
- Binary segmentation: background vs. deck

## Visualization
- Original UAV image
- Predicted deck mask
- Overlay visualization
- Optional comparison with ground truth masks

## Research Use
This repository is intended for:
- UAV-enabled bridge inspection
- Structural health monitoring
- Bridge deck localization
- Preprocessing for crack detection and quantification

---

# Model Performance

| Metric | Value |
|---|---|
| Mean IoU | 97.18% |
| Mean Pixel Accuracy | 98.70% |
| Mean F1 Score | 92.61% |
| Weighted IoU | 97.30% |
| Weighted Pixel Accuracy | 98.63% |

## Class-wise Performance

| Class | Pixel Accuracy | IoU | Boundary F1 Score |
|---|---|---|---|
| Background | 98.35% | 97.75% | 94.54% |
| Deck | 99.05% | 96.61% | 90.68% |

---

# Repository Structure

```text
├── code/
│   ├── DeepLab_Training.m
│   └── inference_deck.m
│
├── sample_dataset_images/
│   ├── images
│   ├── masks
│
├── sample_results/
│   ├── sample_1.png
│   ├── sample_2.png
│   └── ...
│
├── LICENSE
└── README.md
```

---

# Download Pretrained Model

Download the following file from the latest release:

```text
deeplabv3plus_bridge_deck.mat
```

---

# Requirements

## Software
- MATLAB R2025a  
  Earlier R202x releases may also work if the required toolboxes are available.

## Required MATLAB Toolboxes
- Deep Learning Toolbox
- Computer Vision Toolbox
- Image Processing Toolbox
- Parallel Computing Toolbox recommended for GPU inference/training

## Hardware
A CUDA-capable NVIDIA GPU is recommended for efficient training and batch inference.

---

# Dataset

The dataset consists of UAV-acquired bridge inspection images and corresponding pixel-level deck masks.

The segmentation classes are:

```matlab
classNames = ["background", "deck"];
labelIDs   = [0 255];
```

Expected dataset structure:

```text
Dataset/
├── train/
│   ├── images/
│   └── masks/
│
├── validation/
│   ├── images/
│   └── masks/
│
└── test/
    ├── images/
    └── masks/
```

Mask format:
- Background pixels = 0
- Deck pixels = 255

---


# Related Paper

**End-to-End UAV-Enabled Bridge Deck Inspection: From Localization to High-Precision Crack Detection and Quantification**

Accepted for publication in:

**ASCE Journal of Bridge Engineering**

> Note: This repository contains only the bridge deck identification component of the proposed framework. The crack detection and quantification component is maintained separately.

---

# Citation

```bibtex
@article{almasi2026deck,
  title={End-to-End UAV-Enabled Bridge Deck Inspection: From Localization to High-Precision Crack Detection and Quantification},
  author={Almasi, Pouya and Premadasa, Roshira and Jauregui, David and Zhang, Qianyun},
  journal={ASCE Journal of Bridge Engineering},
  note={Accepted for publication},
  year={2026}
}
```

---

# Author

Pouya Almasi  
Ph.D. Candidate in Civil Engineering  
New Mexico State University

Research Areas:
- Structural Health Monitoring
- UAV-based Infrastructure Inspection
- Deep Learning
- Computer Vision
- Bridge Deck Identification
- Semantic Segmentation

---

# License

This project is licensed under the MIT License.

---

# Acknowledgment

The research reported in this work was conducted under a long-term project sponsored by the New Mexico Department of Transportation (NMDOT) Research Bureau. Q. Zhang acknowledges the startup fund from the College of Engineering at New Mexico State University.
