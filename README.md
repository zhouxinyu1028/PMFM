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


Conventional multimodal medical image segmentation methods usually perform
feature fusion directly in voxel space, which may lead to severe feature
entanglement between different imaging modalities and anatomical structures.


To address this problem, PMFM introduces a **virtual point cloud representation**
to transform multimodal feature maps into a unified point-based space.
This representation enables the model to capture latent three-dimensional
spatial relationships while maintaining the computational efficiency of
two-dimensional segmentation networks.


By performing modality–organ decoupling and global semantic interaction
in point cloud space, PMFM effectively improves segmentation accuracy,
boundary consistency, and robustness for complex multi-organ scenarios.


---

# ⭐ Key Contributions


## 1. Virtual Point Cloud Representation


PMFM introduces a virtual point cloud representation that maps multimodal
features into a unified geometric space.


This strategy enables:

- implicit 3D anatomical relationship modeling
- global feature interaction
- reduced modality interference


---

## 2. Point Matrix Module (PMM)


The Point Matrix Module employs PointNet-based global modeling to establish
semantic relationships between:

- different imaging modalities
- different organs
- local and global anatomical structures


---

## 3. Cross Fusion Module (CFM)


The Cross Fusion Module performs deep multimodal feature interaction and
enhances:

- organ boundary representation
- structural consistency
- segmentation robustness


---

## 4. Superior Segmentation Performance


PMFM achieves state-of-the-art performance on the HaN-Seg dataset,
obtaining higher Dice scores and lower Hausdorff distances compared with
mainstream segmentation methods and MICCAI challenge approaches.


---

# 🧠 Network Architecture


The proposed PMFM framework consists of:


- U-Net based segmentation backbone
- Perspective Feature Fusion module
- Point Matrix Module
- Point-cloud feature interaction
- Multimodal fusion decoder


<p align="center">
<img src="figures/network_structure.png" width="90%">
</p>


**Figure 1.**
Overall architecture of PMFM framework.



---

# 📊 Qualitative Results


PMFM demonstrates improved segmentation accuracy compared with representative
methods including:

- U-Net
- UNet++
- nnU-Net


especially for:

- small organs
- irregular anatomical structures
- complex boundaries


<p align="center">
<img src="figures/qualitative_results_new.png" width="90%">
</p>


**Figure 2.**
Qualitative comparison of segmentation results.

From left to right:

Image, Ground Truth, PMFM, nnU-Net, UNet++, and U-Net.


---

# 📈 Quantitative Comparison


| Method | Average Dice (%) | Average HD (mm) |
|---|---:|---:|
| U-Net | 68.30 | 3.84 |
| UNet++ | 64.20 | 7.23 |
| Attention U-Net | 72.50 | 3.38 |
| nnU-Net | 77.50 | 3.26 |
| 2D DDPM | 66.40 | 3.15 |
| Swin U-Net | 69.50 | 4.38 |
| UNETR | 71.90 | 7.23 |
| 3D DDPM | 73.30 | 2.93 |
| CHB-QuantIF | 75.10 | 3.70 |
| UID-Net | 75.20 | 3.90 |
| CWLG102 | 76.80 | 3.80 |
| ELI1 | 76.90 | 3.50 |
| Xie et al. | 77.80 | 3.08 |
| Ren et al. | 77.90 | 3.16 |
| Quetin et al. | 78.10 | 3.45 |
| **PMFM (Ours)** | **79.80** | **2.47** |


PMFM achieves the highest average Dice score and the lowest average HD,
demonstrating superior segmentation accuracy and boundary consistency.


---

# 🔐 Code Availability


The source code of **PMFM** has been publicly released in this repository.


The released implementation includes:


- Perspective Feature Fusion module
- Point Matrix Module
- PointNet-based point cloud representation
- Multimodal feature fusion network
- Loss functions
- Evaluation metrics
- Training pipeline


Project repository:
https://github.com/your_username/PMFM


(Replace the above URL with your official GitHub address.)



---

# 📂 Dataset


Experiments are conducted on the:

**HaN-Seg dataset**


Due to dataset license restrictions,
the original medical images are not redistributed.


Researchers should obtain the dataset from the official HaN-Seg challenge
platform and follow the corresponding access policy.



The expected dataset organization is:

