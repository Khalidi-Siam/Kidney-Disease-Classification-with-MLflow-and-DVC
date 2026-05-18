import streamlit as st
import os
import json
import torch
import numpy as np
from PIL import Image
from kidney_disease_classification.pipeline.stage_07_prediction import PredictionPipeline
from kidney_disease_classification.components.gradcam import GradCAM
from kidney_disease_classification.components.xai import XAI

# Load evaluation metrics
metrics_path = os.path.join("artifacts", "evaluation", "evaluation_report.json")
try:
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
except Exception:
    metrics = {}

@st.cache_resource
def get_prediction_pipeline():
    # Calling the pipeline loads the model
    # It now automatically reads the params bundled inside the best_model.pth!
    return PredictionPipeline()

# Get pipeline and extract its initialized model's true training params
pipeline = get_prediction_pipeline()
params = pipeline.prediction.params 

st.set_page_config(page_title="Kidney Disease Classification", page_icon="🏥", layout="wide")

# Sidebar for Model Info
st.sidebar.title("🧠 Model Information")

if params:
    st.sidebar.markdown("### ⚙️ Training Parameters")
    st.sidebar.info(
        f"**Model:** `{params.get('MODEL_NAME', 'N/A')}`  \n"
        f"**Optimizer:** `{params.get('OPTIMIZER', 'N/A').capitalize()}`  \n"
        f"**Loss:** `{params.get('LOSS_FUNCTION', 'N/A')}`  \n"
        f"**Learning Rate:** `{params.get('LEARNING_RATE', 'N/A')}`  \n"
        f"**Batch Size:** `{params.get('BATCH_SIZE', 'N/A')}`  \n"
        f"**Epochs:** `{params.get('EPOCHS', 'N/A')}`"
    )

if metrics:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Evaluation Metrics")
    m_col1, m_col2 = st.sidebar.columns(2)
    m_col1.metric(label="Accuracy", value=f"{metrics.get('accuracy', 0)*100:.1f}%")
    m_col2.metric(label="F1 Score", value=f"{metrics.get('f1_score', 0):.3f}")
    m_col1.metric(label="Precision", value=f"{metrics.get('precision', 0):.3f}")
    m_col2.metric(label="Recall", value=f"{metrics.get('recall', 0):.3f}")
    m_col1.metric(label="ROC AUC", value=f"{metrics.get('roc_auc', 0):.3f}")

st.title("🏥 Kidney Disease Classification App")
st.markdown("Upload a CT scan image to check if it's **Normal** or has a **Tumor**.")

uploaded_file = st.file_uploader("Choose a CT Scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily for prediction pipeline
    temp_img_path = "temp_uploaded_img.jpg"
    with open(temp_img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Create columns to optimize wide layout space
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        # Display image with an optimized width instead of taking the whole screen
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded CT Scan", width=400)
    
    with col2:
        st.markdown("### 🔍 Analysis & Prediction")
        if st.button("Predict Result🚀", use_container_width=True):
            with st.spinner("Analyzing image and generating XAI heatmaps..."):
                try:
                    pipeline = get_prediction_pipeline()
                    
                    # 1. Normal Prediction Output
                    result = pipeline.main(temp_img_path)
                    
                    pred_label = result["prediction"].capitalize()
                    probability = result["probability"]
                    
                    st.markdown("#### **Results:**")
                    if pred_label.lower() == "disease" or pred_label.lower() == "tumor":
                        st.error(f"Prediction: **Tumor / Disease** ({probability*100:.2f}% probability)")
                    else:
                        st.success(f"Prediction: **Normal** ({(1-probability)*100:.2f}% probability of being normal)")
                        
                    # 2. XAI / Grad-CAM Output
                    st.markdown("#### **Explainable AI (Grad-CAM):**")
                    with st.expander("Show AI Activity Heatmap", expanded=True):
                        # Extract tools from initialized pipeline
                        model = pipeline.prediction.model
                        transform = pipeline.prediction.transform
                        device = pipeline.prediction.device
                        
                        # Apply transform
                        img_tensor = transform(image).unsqueeze(0).to(device)
                        
                        # Initialize GradCAM tracking
                        target_layer = model.features[-1]
                        gradcam = GradCAM(model, target_layer)
                        
                        # Enable gradient tracking temporarily
                        with torch.enable_grad():
                            cam = gradcam.generate_cam(img_tensor)
                            
                        # Overlay heatmap using the XAI static method
                        heatmap_overlay = XAI.overlay_heatmap(image, cam)
                        st.image(heatmap_overlay, caption="Network Activation Heatmap (Red instances indicating high influence)", use_container_width=True)

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

    # Clean up (after the column block so the file is ready to delete)
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
