<img width="1668" height="1435" alt="屏幕截图 2026-02-03 202217" src="https://github.com/user-attachments/assets/3756b9da-1b5c-46d3-90a2-2cf5f3a2a7b4" /># PMFM: Point-cloud Matrix Fusion Model

## Accurate Full Segmentation of Organs-at-risk in Head and Neck Cancer  
### based on Multimodal Point Cloud Fusion

![Status](https://img.shields.io/badge/Status-Under%20Review-orange)
![License](https://img.shields.io/badge/License-TBD-lightgrey)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

---

## 📢 Important Notice / 特别说明

This repository corresponds to a paper that is currently **under review**.  
本文对应论文正处于**投稿 / 审稿阶段**。

To comply with the peer-review policy, the **source code and related resources are not publicly released at this stage**.

---

## 📘 Introduction

This repository corresponds to the paper:

**Accurate Full Segmentation of Organs-at-risk in Head and Neck Cancer based on Multimodal Point Cloud Fusion**

We propose **PMFM**, a novel **Point-cloud Matrix Fusion Model** for accurate and robust segmentation of organs-at-risk (OARs) in head and neck cancer radiotherapy planning.

Unlike conventional voxel-based multimodal fusion methods, PMFM introduces a **virtual point cloud representation** to recover latent three-dimensional anatomical relationships while maintaining the computational efficiency of two-dimensional networks. By performing modality–organ decoupling and global modeling in the point cloud space, PMFM effectively alleviates feature entanglement and improves segmentation robustness in complex multimodal, multi-organ scenarios.

### Key Contributions

- **Virtual Point Cloud Representation**  
  Maps multimodal CT and MRI features into a unified point cloud space to recover latent 3D spatial relationships.

- **Point Matrix Module (PMM)**  
  Employs PointNet-based global modeling to enhance cross-modality and cross-organ semantic associations.

- **Cross Fusion Module (CFM)**  
  Achieves deep inter-modal fusion and improves boundary consistency for complex OAR structures.

- **State-of-the-art Performance**  
  Validated on the HaN-Seg dataset, achieving superior Dice and Hausdorff distance compared with existing methods.

---

## 🧠 Network Architecture

The overall architecture of the proposed **PMFM** framework is illustrated in **Figure 1**.  
The model consists of three core components: the Perspective-based Embedding Module (PEM),  
the Point Matrix Module (PMM), and the Cross Fusion Module (CFM).
<img width="1499" height="960" alt="屏幕截图 2026-02-03 202834" src="https://github.com/user-attachments/assets/05b35ea3-75b5-4bc2-8d87-d634bd4e4da2" />

<p align="center">
  <img src="figures/network_structure.png" width="100%">
</p>

<p align="center">
  <em>
  <b>Figure 1.</b> Overall architecture of the proposed PMFM framework, including the U-Net backbone,
  PEM, PMM, and CFM modules for multimodal point cloud fusion.
  </em>
</p>

---

## 📊 Qualitative Results

**Figure 2** presents qualitative comparisons between the proposed **PMFM** and representative
baseline methods, including nnU-Net, UNet++, and UNet, on head and neck CT slices.

Compared with existing methods, PMFM demonstrates improved boundary consistency and more accurate
segmentation of small and complex organs, particularly in anatomically dense regions.

<img width="1499" height="960" alt="屏幕截图 2026-02-03 202217" src="https://github.com/user-attachments/assets/05b35ea3-75b5-4bc2-8d87-d634bd4e4da2" />

<p align="center">
  <img src="figures/qualitative_results.png" width="100%">
</p>

<p align="center">
  <em>
  <b>Figure 2.</b> Qualitative comparison of segmentation results.
  From left to right: Image, Ground Truth (GT), PMFM, nnU-Net, UNet++, and UNet.
  </em>
</p>

---

## 🔐 Code & Data Availability

- **After paper acceptance**
  - The complete source code will be **fully released in this repository**
  - A **Baidu Netdisk (百度网盘)** link will be provided for datasets, trained models, and supplementary materials

- **During the review stage**
  - Due to copyright and the ongoing peer-review process, the code is currently **password-protected**
  - Access can be granted **for peer review or academic research purposes only**

### Access Procedure

1. **Request access by email**

   Please send an email with the subject:  
   **“Request for PMFM Code”**

   The email should briefly include:
   - Your name and affiliation
   - Purpose of use (e.g., peer review, academic research)

2. **Contact Email**

   📧 **20298326349@qq.com**

   The Baidu Netdisk link and extraction code will be provided upon reasonable request.

---

## 🧩 Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- NumPy  
- SciPy  
- scikit-learn  

(Additional dependencies will be listed after code release.)

---

## 📌 Notes

- This project is intended **for academic research and non-commercial use only**
- Redistribution of the code or data without permission is prohibited
- The repository will be updated promptly after paper acceptance

Thank you for your interest in our work!
