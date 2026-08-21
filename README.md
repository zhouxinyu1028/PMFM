# PMFM: Point-cloud Matrix Fusion Model

## Accurate Full Segmentation of Organs-at-risk in Head and Neck Cancer

### based on Multimodal Point Cloud Fusion

![Status](https://img.shields.io/badge/Status-Code%20Released-007ec6)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C)
![Dataset](https://img.shields.io/badge/Dataset-HaN--Seg-8A2BE2)
![Task](https://img.shields.io/badge/Task-Medical%20Image%20Segmentation-2E8B57)
![Framework](https://img.shields.io/badge/Framework-PyTorch-F7DF1E)

------------------------------------------------------------------------

# Introduction

This repository provides the official implementation of:

**Accurate Full Segmentation of Organs-at-risk in Head and Neck Cancer
based on Multimodal Point Cloud Fusion**

We propose PMFM, a Point-cloud Matrix Fusion Model for accurate and
robust organs-at-risk segmentation in head and neck cancer radiotherapy
planning.

PMFM introduces a virtual point cloud representation to recover latent
three-dimensional anatomical relationships while maintaining the
efficiency of two-dimensional networks.

The framework performs modality-organ decoupling and global semantic
modeling in point cloud space, improving segmentation robustness for
complex multimodal and multi-organ scenarios.

------------------------------------------------------------------------

# Key Contributions

-   Virtual Point Cloud Representation
-   Point Matrix Module (PMM)
-   Cross Fusion Module (CFM)
-   Multimodal feature interaction and boundary refinement

------------------------------------------------------------------------

# Network Architecture

The framework consists of:

-   Perspective-based Embedding Module (PEM)
-   Point Matrix Module (PMM)
-   Cross Fusion Module (CFM)
-   U-Net based segmentation backbone

```{=html}
<p align="center">
```
`<img src="figures/network_structure.png" width="100%">`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

# Quantitative Results

  Method               Dice (%)    HD (mm)
  ----------------- ----------- ----------
  nnU-Net                 77.50       3.26
  Ren et al.              77.90       3.16
  Quetin et al.           78.10       3.45
  **PMFM (Ours)**     **79.80**   **2.47**

PMFM achieves superior segmentation accuracy and boundary consistency.

------------------------------------------------------------------------

# Code & Data Availability

The source code of PMFM has been publicly released.

The released implementation includes:

-   Perspective Feature Fusion module
-   Point Matrix Module
-   Point cloud representation
-   Network construction
-   Loss functions
-   Training pipeline
-   Evaluation metrics

------------------------------------------------------------------------

# Dataset

Experiments are conducted on the HaN-Seg dataset.

Due to dataset license restrictions, the original medical images are not
redistributed.

Dataset organization:

    datasets/
    └── nanseg/
        ├── images/
        └── labels/

Dataset configuration:

    config/nanseg.yaml

------------------------------------------------------------------------

# Requirements

-   Python \>= 3.8
-   PyTorch \>= 1.10

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Usage

## Training

Run:

``` bash
python trainNanSegV4.py
```

Modify training settings in:

    config/nanseg.yaml

## Evaluation

Evaluation metrics are provided in:

    loss/
    └── metrics.py

including:

-   Dice coefficient
-   Hausdorff distance

------------------------------------------------------------------------

# Repository Structure

    PMFM
    │
    ├── config
    │   └── nanseg.yaml
    │
    ├── datasets
    │   └── nanseg
    │       ├── images
    │       └── labels
    │
    ├── loss
    │   ├── loss.py
    │   ├── metrics.py
    │   └── __init__.py
    │
    ├── model
    │   ├── PFF.py
    │   ├── PFFUNet.py
    │   ├── pointnet.py
    │   ├── unet2d.py
    │   ├── builder_model.py
    │   └── __init__.py
    │
    ├── optimizer
    │   ├── builder_optimizer.py
    │   ├── builder_lr_scheduler.py
    │   └── __init__.py
    │
    ├── utils
    │
    ├── trainNanSegV4.py
    │
    ├── requirements.txt
    │
    └── README.md

------------------------------------------------------------------------

# Notes

-   Released for academic research purposes.
-   Dataset is not included due to license restrictions.
-   Please cite our work when using this repository.

------------------------------------------------------------------------

# Citation

``` bibtex
@article{xu2026pmfm,
  title={Accurate full segmentation of organs-at-risk in head and neck cancer based on multimodal point cloud fusion},
  author={Xu, Pengfei and Zhou, Xinyu and Wang, Jie and Liu, Xianyi and Liu, Jinping and Li, Jinxiu and Duan, Xiaohui},
  journal={Medical Image Analysis},
  volume={113},
  pages={104185},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.media.2026.104185}
}
```
