import sys
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
from kidney_disease_classification.entity.config_entity import PredictionConfig
from kidney_disease_classification.exception import CustomException
from kidney_disease_classification.logger import logging


class Prediction:
    def __init__(self, config: PredictionConfig, params):
        self.config = config
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.load_model()
        self.transform = self.get_transform()

    # ---------------------------
    # 1. LOAD MODEL
    # ---------------------------
    def load_model(self):
        try:
            logging.info("Loading trained model for inference...")

            model = models.efficientnet_b0(pretrained=False)

            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, self.params["NUM_CLASSES"])
            
            # Load the checkpoint
            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            
            # Backward compatibility check. See if it's the new packaged checkpoint or old state_dict
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint and "params" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
                # Override the pipeline's params with the one packaged in the model
                self.params = checkpoint["params"] 
                logging.info("Model state and bundled params loaded from checkpoint.")
            else:
                model.load_state_dict(checkpoint)
                logging.info("Legacy model state_dict loaded.")

            model.to(self.device)
            model.eval()

            logging.info("Model loaded successfully for inference.")
            return model

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------------------
    # 2. TRANSFORM (SAME AS TEST)
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
    # 3. PREDICT FUNCTION
    # ---------------------------
    def predict(self, image_path: str):
        try:
            img = Image.open(image_path).convert("RGB")

            input_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor).squeeze()
                prob = torch.sigmoid(output).item()

            pred_label = "disease" if prob > 0.5 else "normal"

            result = {
                "prediction": pred_label,
                "probability": float(prob)
            }

            logging.info(f"Prediction Result: {result}")
            return result

        except Exception as e:
            raise CustomException(e, sys)
        

