# Kidney Disease Classification from CT Scan Images

> An end-to-end MLOps project classifying kidney CT scans as **Normal vs. Tumor** using Deep Learning, full pipeline reproducibility via **DVC**, comprehensive experiment tracking via **MLflow**, and transparent predictions with **Explainable AI (Grad-CAM)** — all served through an interactive **Streamlit** application.

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/DVC-13ADC7?style=for-the-badge&logo=dvc&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
</p>

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [ML Pipeline Architecture](#ml-pipeline-architecture)
- [Project Structure](#project-structure)
- [Model & Hyperparameters](#model--hyperparameters)
- [Evaluation Metrics](#evaluation-metrics)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Explainable AI (Grad-CAM)](#explainable-ai-grad-cam)
- [Getting Started](#getting-started)
- [Running the Application](#running-the-application)
- [Dataset](#dataset)
- [Limitations](#limitations)

---

## Project Overview

Kidney disease diagnosis from CT scan images is a critical clinical challenge. Manual inspection is slow and subject to error. This project builds a fully automated, production-ready classification pipeline that:

- **Classifies** kidney CT scans as either **Normal** or **Tumor** using Transfer Learning
- **Tracks every experiment** (parameters, metrics, artifacts) with MLflow across 3 pipeline stages
- **Ensures reproducibility** with DVC-managed data and pipeline versioning
- **Builds clinical trust** by visualizing model decisions using Grad-CAM heatmaps
- **Surfaces everything** in a modern Streamlit UI for real-time interaction

---

## Key Features

| Feature | Description |
|---|---|
| **Transfer Learning** | `EfficientNet-B0` (also supports `ResNet50`, `DenseNet121`) for binary classification |
| **DVC Pipeline** | 6-stage reproducible pipeline: Ingestion → Validation → Transformation → Training → Evaluation → XAI |
| **MLflow Tracking** | Tracks hyperparameters, loss curves, evaluation metrics, and Grad-CAM artifacts across 3 stages |
| **Explainable AI** | Grad-CAM heatmaps visually highlight which CT scan regions influenced the model's prediction |
| **Data Validation** | Automated dataset integrity checks with JSON validation report before any training begins |
| **Early Stopping** | Configurable patience-based early stopping to prevent overfitting during training |
| **Smart Checkpointing** | Saves best model based on `val_accuracy`; checkpoint bundles weights + training params together |
| **Interactive UI** | Streamlit app with real-time prediction, confidence scores, Grad-CAM overlay, and model metadata panel |
| **Installable Package** | Source code structured as a Python package (`setup.py`) for clean imports across all modules |
| **Research Notebooks** | 6 isolated Jupyter notebooks prototype each pipeline stage independently before modularization |

---

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| **Deep Learning** | PyTorch, Torchvision |
| **MLOps** | DVC, MLflow |
| **XAI** | Grad-CAM (custom), OpenCV |
| **Web App** | Streamlit |
| **Data Processing** | NumPy, Pandas, Pillow, SciPy |
| **Evaluation** | Scikit-learn (Accuracy, Precision, Recall, F1, ROC-AUC, Specificity) |
| **Progress / Logging** | tqdm, Python `logging` (custom logger) |
| **Data Download** | gdown (Google Drive) |
| **Packaging** | setuptools (`setup.py` with `find_packages`) |

---

## ML Pipeline Architecture

The DVC pipeline consists of **6 reproducible stages**, each with explicit dependencies and outputs tracked by DVC:

```
Data Ingestion
      │
      ▼
Data Validation ──► validation_report.json
      │
      ▼
Data Transformation ──► train / val / test splits
      │
      ▼
Training ──────────► best_model.pth ──► MLflow logs: params + loss curves 
      │                                
      ▼                               
Evaluation ────────► evaluation_report.json ──► MLflow logs: accuracy, F1, AUC
      │
      ▼
XAI (Grad-CAM) ────► gradcam_outputs/ ──► MLflow logs: heatmap artifacts
```

Run the complete pipeline end-to-end with a single command:
```bash
dvc repro
```

DVC only re-runs stages whose dependencies have changed, making iteration fast and auditable.

---

## Project Structure

```text
Kidney-Disease-Classification-with-MLflow-and-DVC/
│
├── app.py                          # Streamlit web application (prediction + XAI UI)
├── main.py                         # Manual pipeline execution entry point
├── dvc.yaml                        # DVC pipeline stage definitions
├── dvc.lock                        # DVC lock file (reproducibility snapshot)
├── params.yaml                     # All model hyperparameters (single source of truth)
├── setup.py                        # Package installation config
├── requirements.txt                # Python dependencies
│
├── config/
│   └── config.yaml                 # All file paths, MLflow URIs, experiment names
│
├── research/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_data_transformation.ipynb
│   ├── 04_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_xAI.ipynb                # Prototype notebooks for each pipeline stage
│
├── src/kidney_disease_classification/
│   ├── components/                  # Core ML logic (one component per stage)
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   ├── gradcam.py               # GradCAM class
│   │   ├── xai.py                   # XAI heatmap generation & overlay
│   │   └── prediction.py            # Inference logic
│   │
│   ├── pipeline/                    # Stage orchestration (called by DVC & main.py)
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_data_validation.py
│   │   ├── stage_03_data_transformation.py
│   │   ├── stage_04_training.py
│   │   ├── stage_05_evaluation.py
│   │   ├── stage_06_xai.py
│   │   └── stage_07_prediction.py   # Used by Streamlit app
│   │
│   ├── config/                      # Config manager (reads config.yaml)
│   ├── entity/                      # Dataclass config entities per stage
│   ├── constants/                   # Project-wide constants
│   ├── utils/                       # Shared utilities (save_json, load_yaml, etc.)
│   ├── logger.py                    # Custom logging setup
│   └── exception.py                 # Custom exception with traceback
│
├── artifacts/                       # DVC-managed pipeline outputs (auto-generated)
│   ├── data_ingestion/
│   ├── data_validation/             # status.txt + validation_report.json
│   ├── data_transformation/         # train / val / test splits
│   ├── training/model/best_model.pth
│   ├── evaluation/                  # evaluation_report.json + confusion_matrix.json
│   └── xai/gradcam_outputs/         # Grad-CAM heatmap images
│
├── mlartifacts/                     # MLflow artifact store (auto-generated)
├── mlflow.db                        # MLflow SQLite backend
└── logs/                            # Runtime log files
```

---

## Model & Hyperparameters

All hyperparameters live in [`params.yaml`](params.yaml) — the **single source of truth** consumed by DVC and MLflow:

| Parameter | Value | Description |
|---|---|---|
| `MODEL_NAME` | `efficientnet_b0` | Backbone model (`resnet50`, `densenet121` also supported) |
| `PRETRAINED` | `True` | Use ImageNet pre-trained weights |
| `IMAGE_SIZE` | `[224, 224]` | Input resolution for all images |
| `BATCH_SIZE` | `16` | Training & evaluation batch size |
| `EPOCHS` | `10` | Maximum training epochs |
| `OPTIMIZER` | `adam` | Optimizer algorithm |
| `LEARNING_RATE` | `0.0001` | Initial learning rate |
| `WEIGHT_DECAY` | `0.00001` | L2 regularization factor |
| `LOSS_FUNCTION` | `bcewithlogits` | Binary Cross-Entropy with Logits (binary classification) |
| `NUM_CLASSES` | `1` | Single sigmoid output for binary classification |
| `AUGMENTATION` | `True` | Apply data augmentation during transformation |
| `TRAIN_SPLIT` | `0.70` | 70% of data for training |
| `VAL_SPLIT` | `0.15` | 15% of data for validation |
| `TEST_SPLIT` | `0.15` | 15% of data for testing |
| `SEED` | `42` | Random seed for reproducibility |

> **Early Stopping** is enabled (`patience=5`) monitoring `val_accuracy`. The best checkpoint is automatically saved.

---

## Evaluation Metrics

The evaluation stage computes and logs the following metrics to both a JSON report and MLflow:

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted tumors, how many were actually tumors |
| **Recall (Sensitivity)** | Of actual tumors, how many were correctly detected |
| **Specificity** | Of actual normal cases, how many were correctly identified |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under the ROC curve |
| **Confusion Matrix** | TP, TN, FP, FN breakdown (saved separately) |

Metrics are saved to `artifacts/evaluation/evaluation_report.json` and displayed live in the Streamlit sidebar.

---

## MLflow Experiment Tracking

MLflow tracks every run under the experiment name **`Kidney_Disease_Classification`** across **3 pipeline stages**:

| Stage | What Is Logged |
|---|---|
| **Training** | Hyperparameters (model, optimizer, LR, epochs, batch size), train/validation loss per epoch |
| **Evaluation** | Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, confusion matrix artifact |
| **XAI** | Generated Grad-CAM heatmap images as artifacts |

Launch the MLflow tracking UI:
```bash
mlflow ui
```
Then navigate to **`http://localhost:5000`** to compare runs, inspect metrics, and download artifacts.

---

## Explainable AI (Grad-CAM)

Medical AI demands transparency. This project integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to answer *"Why did the model predict this?"*

- **How it works:** Grad-CAM computes gradients of the final convolutional layer's activations with respect to the predicted class, producing a heatmap over the original image.
- **Target Layer:** `model.features[-1]` (last convolutional block of EfficientNet-B0)
- **Pipeline Stage:** Stage 6 generates heatmaps for 20 test samples, logs them to MLflow
- **Live in App:** The Streamlit UI generates Grad-CAM overlays **in real-time** for any uploaded CT scan

Red/warm regions in the heatmap indicate areas the model focused on most heavily when making its prediction.

---

## Getting Started

### Prerequisites
- Python 3.8+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Khalidi-Siam/Kidney-Disease-Classification-with-MLflow-and-DVC.git
cd Kidney-Disease-Classification-with-MLflow-and-DVC
```

### 2. Create & Activate a Virtual Environment
```bash
python -m venv env

# Windows
env\Scripts\activate

# macOS / Linux
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
> This also installs the `kidney_disease_classification` package in editable mode via `-e .`

### 4. Reproduce the Full ML Pipeline
DVC will automatically download the dataset, run all 6 stages, and generate all artifacts:
```bash
dvc repro
```
> Only stages with changed dependencies are re-executed on subsequent runs.

---

## Running the Application

### Launch the Streamlit App
```bash
streamlit run app.py
```

The app provides:
- **Upload** any kidney CT scan (`.jpg`, `.jpeg`, `.png`)
- **Prediction** with confidence score (Normal / Tumor)
- **Grad-CAM heatmap** overlay showing model attention areas
- **Model information sidebar** — training parameters + evaluation metrics

### View MLflow Experiments
```bash
mlflow ui
```
Access at `http://localhost:5000`

---

## Dataset

The dataset used is the **CT Kidney Dataset** (binary subset: Normal vs. Tumor).

- **Source:** Hosted on Google Drive, auto-downloaded during the Data Ingestion stage
- **Download URL:** Configured in `config/config.yaml` (`source_URL`)
- **Format:** Organized as `kidney-ct-scan-image/Normal/` and `kidney-ct-scan-image/Tumor/` directories
- **Splits:** Automatically split 70/15/15 (train/val/test) during the Data Transformation stage

---

## Limitations

| Limitation | Details |
|---|---|
| **Small Dataset** | The dataset contains a limited number of CT scan samples. A larger, more diverse dataset would significantly improve model generalization to unseen real-world cases. |
| **Binary Classification Only** | The current model classifies only **Normal vs. Tumor**. The original CT Kidney dataset includes Cyst and Stone categories, which are not modeled here. |
| **Single Modality** | The model is trained solely on CT scan images. Clinical diagnosis typically incorporates additional data (patient history, blood markers, etc.) not considered here. |
| **Not Clinically Validated** | This project is built for educational and demonstration purposes. It has **not** been validated against clinical standards and should **not** be used for actual medical diagnosis. |

---
*Built to demonstrate robust MLOps practices, Interpretable Deep Learning, and production-grade medical imaging pipelines.*
