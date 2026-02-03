# PMFM: Point-cloud Matrix Fusion Model

## Accurate Full Segmentation of Organs-at-risk in Head and Neck Cancer  
### based on Multimodal Point Cloud Fusion

![Status](https://img.shields.io/badge/Status-Under%20Review-orange)
![License](https://img.shields.io/badge/License-TBD-lightgrey)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

---

## 📢 Important Notice / 特别说明

This paper is currently **under review**.  
本文对应论文正处于**投稿 / 审稿阶段**。

To comply with the peer-review policy, the **source code and related resources are not publicly released at this time**.

---

## 📘 Introduction

This repository corresponds to the paper:

**Accurate Full Segmentation of Organs-at-risk in Head and Neck Cancer based on Multimodal Point Cloud Fusion**

We propose **PMFM**, a novel **Point-cloud Matrix Fusion Model** for accurate and robust segmentation of organs-at-risk (OARs) in head and neck cancer radiotherapy planning.

PMFM leverages a **virtual point cloud representation** to model latent three-dimensional anatomical relationships while maintaining the computational efficiency of 2D networks. By decoupling modalities and organs at the point level and performing global modeling via point cloud attention, PMFM effectively alleviates feature entanglement and improves segmentation accuracy in complex multimodal, multi-organ scenarios.

### Key Contributions

- **Point-cloud-based Multimodal Fusion**  
  Introduces a virtual point cloud representation to unify CT and MRI features and recover latent 3D spatial relationships.

- **Point Cloud Matrix Module (PMM)**  
  Employs PointNet-based global modeling to enhance cross-modality and cross-organ semantic associations.

- **Cross Fusion Module (CFM)**  
  Achieves deep inter-modal fusion and improves boundary consistency and segmentation robustness.

- **State-of-the-art Performance**  
  Validated on the HaN-Seg dataset, achieving superior Dice and Hausdorff distance compared with existing methods.

---

## 🔐 Code & Data Availability

- **After paper acceptance**  
  - The complete source code will be **fully released in this repository**
  - A **Baidu Netdisk (百度网盘)** link will be provided for datasets, trained models, and supplementary materials

- **During the review stage**  
  - Due to copyright and the ongoing peer-review process, the code is currently **password-protected**
  - Researchers may request access **for peer review or academic research purposes**

### Access Procedure

1. **Request Access by Email**

   Please send an email with the subject:  
   **“Request for PMFM Code”**

   Email content should briefly include:
   - Your name and institution
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
- (Other dependencies will be listed after code release)

---

## 📌 Notes

- This project is intended **for academic research and non-commercial use only**
- Please do not redistribute any part of the code or data without permission
- The repository will be updated promptly after paper acceptance

Thank you for your interest in our work!
