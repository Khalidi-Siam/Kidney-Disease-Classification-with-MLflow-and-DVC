import streamlit as st
import os
from kidney_disease_classification.utils.common import read_yaml
from PIL import Image
from kidney_disease_classification.pipeline.stage_07_prediction import PredictionPipeline


@st.cache_resource
def get_prediction_pipeline():
    return PredictionPipeline()

st.set_page_config(page_title="Kidney Disease Classification", page_icon="🏥", layout="centered")

st.title("🏥 Kidney Disease Classification App")
st.markdown("Upload a CT scan image to check if it's **Normal** or has a **Tumor**.")

uploaded_file = st.file_uploader("Choose a CT Scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded CT Scan", use_container_width=True)
    
    # Save uploaded file temporarily for prediction pipeline
    temp_img_path = "temp_uploaded_img.jpg"
    with open(temp_img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            try:
                pipeline = get_prediction_pipeline()
                result = pipeline.main(temp_img_path)
                
                pred_label = result["prediction"].capitalize()
                probability = result["probability"]
                
                if pred_label.lower() == "disease" or pred_label.lower() == "tumor":
                    st.error(f"Prediction: **Tumor / Disease** ({probability*100:.2f}% probability)")
                else:
                    st.success(f"Prediction: **Normal** ({(1-probability)*100:.2f}% probability of being normal)")
                
                # Clean up
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                    
            except Exception as e:
                st.error(f"Error during prediction: {e}")
