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

### 1. Prerequisites
* **Linux Machine** with NVIDIA GPU.
* **Docker Engine** & **Docker Compose** installed.
* **NVIDIA Container Toolkit** (for GPU support in Docker).

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/aerial-instance-segmentation.git
cd satellite-building-instance-segmentation
```

### 3. Installation
1.  **Unzip the project** (or clone the repository).
2.  **Verify the Environment File**: Ensure the `.env` file containing the API keys is present in the root directory.
3.  **Start the Environment**:
    Run the following command to build the Docker container and start the runtime. This may take a few minutes as it compiles the optimized environment.
    ```bash
    docker compose up -d --build
    ```

### 4. Run the Pipeline
Once the container is running (check with `docker ps`), execute the following commands in order:

#### Step A: Data Preparation
Downloads the dataset and merges it with custom labeled data.
```bash
docker exec -it aerial_segmentation_runtime python -m src.stages.data_prep
```

#### Step B: Training

Trains the model based on the configuration in `params.yaml`.

* **Default**: Custom Architecture (ResNet-like + CBAM + GroupNorm).
* **Action**: Trains for the configured epochs, logs metrics (mAP, IoU, Dice) to Weights & Biases, and saves the best model.
* **Transfer Learning**: Automatically handles a 2-phase "Freeze -> Fine-Tune" schedule if selected.



```bash
docker exec -it aerial_segmentation_runtime python -m src.stages.train

```

#### Step C: Inference & Evaluation

Runs the trained model on the unseen Test Set to visualize performance.

* **Action**: Generates visual comparisons showing: [ Original Image | Ground Truth | Prediction ].


* **Output**: Images are saved to the `runs/inference_results/` folder on your host machine.


```bash
docker exec -it aerial_segmentation_runtime python -m src.stages.inference

```


## ⚙️ Configuration

You can switch between the **Custom Architecture** and **Transfer Learning** models by editing `params.yaml`:

```yaml
train:
  # OPTIONS: "custom" OR "transfer"
  model_type: "custom"

```

After changing this setting, simply re-run **Step B (Training)** to train the new model and **Step C (Inference)** to see its results.


