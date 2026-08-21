# PMFM: Point-cloud Matrix Fusion Model

## Accurate Full Segmentation of Organs-at-risk in Head and Neck Cancer  
### based on Multimodal Point Cloud Fusion


![Status](https://img.shields.io/badge/Status-Code%20Released-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)



---

# 📘 Introduction


This repository provides the official implementation of:

**Accurate Full Segmentation of Organs-at-risk in Head and Neck Cancer based on Multimodal Point Cloud Fusion**


We propose **PMFM (Point-cloud Matrix Fusion Model)**,
a novel multimodal segmentation framework for accurate and robust
organs-at-risk (OARs) segmentation in head and neck cancer radiotherapy planning.


Unlike conventional voxel-based multimodal fusion strategies,
PMFM introduces a **virtual point cloud representation** to model
latent three-dimensional anatomical relationships while maintaining
the computational efficiency of 2D convolutional networks.


By performing modality–organ decoupling and global semantic modeling
in point cloud space, PMFM effectively reduces feature entanglement
and improves segmentation performance in complex multimodal and
multi-organ scenarios.



---

# ⭐ Key Contributions


## 1. Virtual Point Cloud Representation

PMFM maps multimodal CT/MR feature representations into a unified
point cloud space to capture latent 3D anatomical structures.


## 2. Point Matrix Module (PMM)

A PointNet-based global modeling strategy is introduced to enhance
cross-modality and cross-organ semantic interaction.


## 3. Cross Fusion Module (CFM)

A feature fusion mechanism is designed to improve boundary consistency
and structural representation of small and complex organs.


## 4. Superior Segmentation Performance

PMFM achieves competitive performance on the HaN-Seg dataset,
with improved Dice coefficient and Hausdorff distance compared with
existing methods.



---

# 🧠 Network Architecture


The proposed PMFM framework consists of:

- Perspective-based Feature Embedding
- Point Matrix Modeling
- Point-cloud Feature Fusion
- U-Net based segmentation decoder


<p align="center">
<img src="figures/network_structure.png" width="90%">
</p>


**Figure 1.**
Overall architecture of PMFM, including feature embedding,
point matrix modeling, point cloud fusion and segmentation network.



---

# 📊 Qualitative Results


PMFM provides more accurate segmentation results compared with
representative methods including U-Net, UNet++, and nnU-Net.


<p align="center">
<img src="figures/qualitative_results_new.png" width="90%">
</p>


**Figure 2.**
Qualitative comparison of segmentation results.
From left to right:

Image, Ground Truth, PMFM, nnU-Net, UNet++, and U-Net.



---

# 📈 Quantitative Comparison


| Methods / Team | Average Dice (%) | Average HD (mm) |
|---|---:|---:|
| U-Net | 68.30 | 3.84 |
| UNet++ | 64.20 | 7.23 |
| Attention U-Net | 72.50 | 3.38 |
| nnU-Net | 77.50 | 3.26 |
| UNETR | 71.90 | 7.23 |
| 3D DDPM | 73.30 | 2.93 |
| Xie et al. | 77.80 | 3.08 |
| Ren et al. | 77.90 | 3.16 |
| Quetin et al. | 78.10 | 3.45 |
| **PMFM (Ours)** | **79.80** | **2.47** |


PMFM achieves the highest average Dice score and the lowest Hausdorff
distance, demonstrating improved segmentation accuracy and boundary
consistency.



---

# 🔐 Code & Data Availability


The complete source code of **PMFM** has been publicly released.


The released implementation contains:


- Perspective Feature Fusion module
- Point Matrix Module
- PointNet-based point cloud representation
- Multimodal feature fusion network
- Loss functions
- Evaluation metrics
- Training pipeline



The code is available at:

