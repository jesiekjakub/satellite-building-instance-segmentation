# 🛰️ Aerial Building Instance Segmentation

An end-to-end Computer Vision pipeline designed to extract individual building footprints from high-resolution satellite imagery. This project implements a lot of state-of-the-art solution to handle dense urban environments and class imbalance.

## 🎯 Objective
The primary goal is **Instance Segmentation**—detecting each building and generating a precise pixel-level mask. To improve prediction quality in dense areas, we solve the additional problem of **Boundary Refinement** using a composite loss function (Focal Loss + Dice Loss) to ensure sharp building edges and robust performance on small objects.

---

## 📂 Dataset
We utilize the [Satellite Building Segmentation Dataset](https://huggingface.co/datasets/keremberke/satellite-building-segmentation).
* [cite_start]**Scale:** almost 10,000 high-resolution satellite images.
* [cite_start]**Custom Data:** Includes an additional 500+ hand-annotated samples to improve geographic diversity.



---

## 🛠️ Tech Stack & MLOps Integration
This project is built with a focus on reproducibility and experiment transparency:

* **Experiment Tracking:** [Weights & Biases](https://wandb.ai) for logging training dynamics, including Validation IoU, Gradient Norms, and Learning Rate decay.
* **Data Versioning:** [DVC](https://dvc.org) to manage large-scale imagery and mask metadata.
* **Environment:** Docker containerization for a hardware-agnostic runtime.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/aerial-instance-segmentation.git
cd satellite-building-instance-segmentation
