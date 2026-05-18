import os
import sys
import json
import random
import torch
import torch.nn as nn
import numpy as np
import cv2
import mlflow
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
from kidney_disease_classification.entity.config_entity import XAIConfig
from .gradcam import GradCAM
from kidney_disease_classification.exception import CustomException
from kidney_disease_classification.logger import logging



class XAI:
    def __init__(self, config: XAIConfig, params):
        self.config = config
        self.params = params

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # 1. LOAD MODEL
    # ---------------------------
    def load_model(self):
        try:
            model = models.efficientnet_b0(pretrained=False)

            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, self.params["NUM_CLASSES"])

            model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()

            logging.info("Model loaded successfully for XAI.")
            return model

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------------------
    # 2. GET TRANSFORM
    # ---------------------------
    def get_transform(self):
        img_size = tuple(self.params["IMAGE_SIZE"])

        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        return transform

    # ---------------------------
    # 3. LOAD RANDOM TEST IMAGES
    # ---------------------------
    def get_sample_images(self):
        test_dir = Path(self.config.dataset_dir) / "test"

        all_images = list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.png")) + list(test_dir.rglob("*.jpeg"))

        if len(all_images) == 0:
            raise CustomException(f"No images found in {test_dir}", sys)

        random.shuffle(all_images)

        num_samples = min(self.config.num_samples, len(all_images))
        return all_images[:num_samples]

    # ---------------------------
    # 4. OVERLAY HEATMAP
    # ---------------------------
    def overlay_heatmap(self, original_img, cam):
        original_img = np.array(original_img)

        if len(original_img.shape) == 2:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

        cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

        return overlay

    # ---------------------------
    # 5. MAIN FUNCTION
    # ---------------------------
    def generate_gradcam_outputs(self):
        try:
            logging.info("Starting Grad-CAM XAI stage...")

            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment_name)


            with mlflow.start_run(run_name="xai_run"):
                model = self.load_model()
                transform = self.get_transform()

                # EfficientNet target conv layer
                target_layer = model.features[-1]

                gradcam = GradCAM(model, target_layer)

                sample_images = self.get_sample_images()

                normal_count = 0
                disease_count = 0

                for img_path in sample_images:
                    img = Image.open(img_path).convert("RGB")
                    input_tensor = transform(img).unsqueeze(0).to(self.device)

                    output = model(input_tensor).squeeze()
                    
                    prob = torch.sigmoid(output).item()
                    pred_label = "disease" if prob > 0.5 else "normal"

                    if pred_label == "disease":
                        disease_count += 1
                    else:
                        normal_count += 1

                    cam = gradcam.generate_cam(input_tensor)

                    overlay = self.overlay_heatmap(img, cam)

                    save_path = os.path.join(
                        self.config.output_dir,
                        f"{img_path.stem}_pred_{pred_label}_prob_{prob:.2f}.png"
                    )

                    Image.fromarray(overlay).save(save_path)
                    mlflow.log_artifact(save_path, artifact_path="gradcam_outputs")
                    logging.info(f"Saved Grad-CAM: {save_path}")

                summary = {
                    "total_images": len(sample_images),
                    "output_dir": self.config.output_dir
                }

                summary_path = os.path.join(self.config.output_dir, "xai_summary.json")

                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=4)

                mlflow.log_artifact(summary_path, artifact_path="xai_summary")

                mlflow.log_metrics({
                    "normal_predictions": normal_count,
                    "disease_predictions": disease_count
                })

            logging.info("Grad-CAM XAI stage completed successfully.")
            return True

        except Exception as e:
            raise CustomException(e, sys)