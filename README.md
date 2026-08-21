# PMFM: Point-cloud Matrix Fusion Model

## Accurate Full Segmentation of Organs-at-risk in Head and Neck Cancer
### Based on Multimodal Point Cloud Fusion


![Status](https://img.shields.io/badge/Status-Code%20Released-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)



# 📘 Introduction


This repository provides the official implementation of:


**Accurate Full Segmentation of Organs-at-risk in Head and Neck Cancer based on Multimodal Point Cloud Fusion**


We propose **PMFM (Point-cloud Matrix Fusion Model)**,
a novel multimodal segmentation framework designed for accurate
and robust segmentation of organs-at-risk (OARs) in head and neck cancer radiotherapy planning.


Unlike conventional voxel-based multimodal fusion strategies,
PMFM introduces a virtual point cloud representation to model latent
three-dimensional anatomical relationships while maintaining the efficiency
of 2D convolutional architectures.


By performing modality-organ decoupling and global semantic interaction
in point cloud space, PMFM effectively reduces feature entanglement
and improves segmentation accuracy in complex multimodal and multi-organ scenarios.



# ⭐ Highlights


## Virtual Point Cloud Representation

PMFM transforms multimodal image features into a unified virtual point cloud space,
allowing the network to capture hidden 3D anatomical relationships.


## Point Matrix Module (PMM)

A PointNet-based global modeling module is introduced to establish
cross-modality and cross-organ semantic correlations.


## Cross Fusion Module (CFM)

A dedicated fusion module is designed to achieve deep interaction
between modalities and improve boundary prediction.



## State-of-the-art Performance

Evaluated on the HaN-Seg dataset,
PMFM achieves superior segmentation performance compared with
mainstream segmentation networks and MICCAI challenge methods.



---

# 🧠 Network Architecture


The PMFM framework consists of three major components:


- Perspective-based Embedding Module (PEM)
- Point Matrix Module (PMM)
- Cross Fusion Module (CFM)



<p align="center">
<img src="figures/network_structure.png" width="100%">
</p>


**Figure 1.**
Overall architecture of PMFM framework.



---

# 📊 Experimental Results


## Qualitative Comparison


<p align="center">
<img src="figures/qualitative_results_new.png" width="100%">
</p>



PMFM demonstrates improved boundary consistency and better
segmentation capability for small and anatomically complex organs.



---

## Quantitative Comparison


| Method | Dice (%) | HD (mm) |
|---|---:|---:|
| U-Net |68.30|3.84|
| UNet++|64.20|7.23|
| Attention U-Net|72.50|3.38|
| nnU-Net|77.50|3.26|
| Swin U-Net|69.50|4.38|
| UNETR|71.90|7.23|
| 3D DDPM|73.30|2.93|
| Ren et al.|77.90|3.16|
| Quetin et al.|78.10|3.45|
| **PMFM (Ours)**|**79.80**|**2.47**|



---

# 🔐 Code and Model Availability


The complete source code of PMFM is publicly released in this repository.



The released code includes:


- Network architecture
- PEM module
- PMM module
- CFM module
- Loss functions
- Training pipeline
- Evaluation pipeline
- Configuration files



---

# 📂 Dataset


Experiments are conducted on the:


**HaN-Seg: Head and Neck Organ-at-Risk Segmentation Dataset**


Due to dataset license restrictions,
the original medical images are not redistributed.


Users should obtain the dataset from the official HaN-Seg platform
according to the dataset policy.


