# PMFM: Point-cloud Matrix Fusion Model

## Accurate Full Segmentation of Organs-at-risk in Head and Neck Cancer  
### based on Multimodal Point Cloud Fusion


![Status](https://img.shields.io/badge/Status-Code%20Released-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)


---

# 📘 Introduction

This repository corresponds to the paper:

**Accurate Full Segmentation of Organs-at-risk in Head and Neck Cancer based on Multimodal Point Cloud Fusion**

We propose **PMFM**, a novel **Point-cloud Matrix Fusion Model** for accurate and robust segmentation of organs-at-risk (OARs) in head and neck cancer radiotherapy planning.

Unlike conventional voxel-based multimodal fusion methods, PMFM introduces a **virtual point cloud representation** to recover latent three-dimensional anatomical relationships while maintaining the computational efficiency of two-dimensional networks.

By performing modality–organ decoupling and global modeling in the point cloud space, PMFM effectively alleviates feature entanglement and improves segmentation robustness in complex multimodal, multi-organ scenarios.


## Key Contributions

- **Virtual Point Cloud Representation**

  Maps multimodal CT and MR features into a unified point cloud space to recover latent 3D spatial relationships.

- **Point Matrix Module (PMM)**

  Employs PointNet-based global modeling to enhance cross-modality and cross-organ semantic associations.

- **Cross Fusion Module (CFM)**

  Achieves deep inter-modal fusion and improves boundary consistency for complex OAR structures.

- **State-of-the-art Performance**

  Validated on the HaN-Seg dataset, achieving superior Dice coefficient and Hausdorff distance compared with existing methods.



---

# 🧠 Network Architecture

The overall architecture of the proposed **PMFM** framework is illustrated in Figure 1.

The model consists of three core components:

- Perspective-based Embedding Module (PEM)
- Point Matrix Module (PMM)
- Cross Fusion Module (CFM)


<img width="1499" height="960" alt="Network Architecture" src="https://github.com/user-attachments/assets/05b35ea3-75b5-4bc2-8d87-d634bd4e4da2" />


<p align="center">
  <img src="figures/network_structure.png" width="100%">
</p>


<p align="center">
<b>Figure 1.</b>
Overall architecture of the proposed PMFM framework, including the U-Net backbone,
PEM, PMM, and CFM modules for multimodal point cloud fusion.
</p>



---

# 📊 Qualitative Results

Figure 2 presents qualitative comparisons between the proposed **PMFM**
and representative baseline methods, including nnU-Net, UNet++, and UNet,
on head and neck CT slices.

Compared with existing methods, PMFM demonstrates improved boundary consistency
and more accurate segmentation of small and complex organs, particularly in
anatomically dense regions.


<img width="1705" height="1435" alt="Qualitative Results" src="https://github.com/user-attachments/assets/1c69633d-ebcd-4f8a-8215-f601b58f536d" />


<p align="center">
<img src="figures/qualitative_results_new.png" width="100%">
</p>


<p align="center">
<b>Figure 2.</b>
Qualitative comparison of segmentation results.
From left to right:
Image, Ground Truth (GT), PMFM, nnU-Net, UNet++, and UNet.
</p>



---

# 📈 Quantitative Comparison with Mainstream and MICCAI Challenge Methods


To further evaluate the effectiveness of PMFM,
we compare it with representative mainstream deep learning methods
and MICCAI HaN-Seg challenge methods.

<div align="center">


| Methods / Team        | Average Dice (%) | Average HD (mm) |
|-----------------------|------------------|-----------------|
| U-Net                 | 68.30            | 3.84            |
| UNet++                | 64.20            | 7.23            |
| Attention U-Net       | 72.50            | 3.38            |
| nnU-Net               | 77.50            | 3.26            |
| 2D DDPM               | 66.40            | 3.15            |
| Salmanpour et al.     | 67.70            | 5.42            |
| Swin U-Net            | 69.50            | 4.38            |
| UNETR                 | 71.90            | 7.23            |
| 3D DDPM               | 73.30            | 2.93            |
| CHB-QuantIF           | 75.10            | 3.70            |
| UID-Net               | 75.20            | 3.90            |
| CWLG102               | 76.80            | 3.80            |
| ELI1                  | 76.90            | 3.50            |
| Xie et al.            | 77.80            | 3.08            |
| Ren et al.            | 77.90            | 3.16            |
| Quetin et al.         | 78.10            | 3.45            |
| **PMFM (Ours)**       | **79.80**        | **2.47**        |

</div>


**Table 1.**
Quantitative comparison between PMFM and mainstream segmentation methods
as well as MICCAI HaN-Seg challenge methods.

PMFM achieves the highest average Dice score and the lowest average HD,
demonstrating superior segmentation accuracy and boundary consistency.



---

# 🔐 Code & Data Availability


The source code of **PMFM** has been publicly released in this repository.


The released implementation includes:


- Perspective-based Embedding Module (PEM)
- Point Matrix Module (PMM)
- Cross Fusion Module (CFM)
- Network construction
- Loss functions
- Training and evaluation pipeline



## Dataset


Experiments are conducted on the **HaN-Seg dataset**.


Due to dataset license restrictions,
the original medical images are not redistributed in this repository.


Researchers should obtain the dataset from the official HaN-Seg challenge platform
and follow the corresponding access policy.



## Pre-trained Models


The pretrained weights are not included in the current release.


Researchers who require pretrained models for reproduction or comparison
may contact:


📧 **20298326349@qq.com**



---

# 🧩 Requirements


- Python ≥ 3.8
- PyTorch ≥ 1.10


Install dependencies:


```bash
pip install -r requirements.txt
```



---

# 🚀 Usage


## 1. Dataset Preparation


Prepare the HaN-Seg dataset:


```
datasets/
└── hanseg/
    ├── images/
    └── labels/
```


Modify the dataset path in:


```
config/hanseg.yaml
```



---

## 2. Training


Run:


```bash
python train.py --config config/hanseg.yaml
```



---

## 3. Evaluation


Run:


```bash
python test.py --config config/hanseg.yaml
```



---

# 📂 Repository Structure


```
PMFM
│
├── config
│   └── hanseg.yaml
│
├── datasets
│
├── loss
│
├── model
│   ├── CFM.py
│   ├── PMFM.py
│   ├── pointnet.py
│   └── unet2d.py
│
├── utils
│
├── train.py
├── test.py
├── requirements.txt
│
└── README.md

```



---

# 📌 Notes


- This project is released for academic research and non-commercial use only.
- Redistribution of the dataset is prohibited.
- Please cite our work when using this code.
- Issues and suggestions are welcome.



---

# 📚 Citation


If you use this repository in your research,
please cite:


```bibtex
@article{
PMFM,
title={Accurate Full Segmentation of Organs-at-risk in Head and Neck Cancer based on Multimodal Point Cloud Fusion},
author={},
journal={},
year={}
}
```



Thank you for your interest in our work!
