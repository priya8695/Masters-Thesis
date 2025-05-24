

# Automated Aorta Segmentation and Geometric Analysis for TEVAR Planning

This repository contains the code and research artifacts for the Master's thesis titled:

**"Automated Aorta Segmentation and Geometric Analysis using Deep Learning for Endovascular Surgical Planning"**

🧠 Focus: Deep Learning for **CT-based Aorta Segmentation** and **Geometric Feature Extraction** relevant to **TEVAR** (Thoracic Endovascular Aortic Repair).

---
# Introduction
Endovascular repair of the thoracic aorta, also referred to as thoracic endovascular aortic repair (TEVAR), refers to a minimally invasive approach that involves placing a stent-graft in the thoracic or thoracoabdominal aorta for the treatment of a variety of thoracic aortic pathologies. In contradiction to open surgery, TEVAR results in reduced recovery times and potentially improved survival rates. Feasibility of TEVAR and correct endograft sizing are based on measurements of Ishimaru’s proximal landing zones. 
However, TEVAR of the aortic arch still carries a significant risk of medium and long-term complications, including endoleak, endograft migration, and collapse.
This may be due to its complex structure and computation of geometric parameters, such as angulation and tortuosity can help to avoid hostile landing zones. 
The primary goal of this project is to segment the aorta from provided CT scan images, map the landzones (Ishimaru’s proximal landing zones Z0, Z1, Z2, and Z3), and compute various geometric parameters of the aortic arch. 
---

## 🎯 Objectives

- Implement and evaluate deep learning models (3D U-Net, UNETR) for aorta segmentation.
- Automate geometric analysis including:
  - **Tortuosity Index**
  - **Tortuosity Angle**
  - **Segment Length**
  - **Diameter**
- Validate segmentation and measurements against manual annotations.

---

## 🧪 Methods

- **Input**: Thoracic CT scans
- **Segmentation Models**:
  - 3D U-Net
  - UNETR (Transformer-based)
- **Preprocessing**:
  - Normalization
  - Data Augmentation: Elastic deformation, Affine, Noise, Gamma, Flip
- **Postprocessing**:
  - Proximal Landing Zone (PLZ) Mapping
  - Geometric Parameter Extraction

---

## 📊 Results

| Model     | Dice Score |
|-----------|------------|
| 3D U-Net  | 0.931      |
| UNETR     | 0.897      |

- **Average error** for most parameters: **< 10%**
- **Tortuosity Index error**: 18.78%
- Validated PLZ mapping on 3 subjects (standard + CILCA arch)

---


