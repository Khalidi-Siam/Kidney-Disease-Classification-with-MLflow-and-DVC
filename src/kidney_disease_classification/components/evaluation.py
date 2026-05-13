import sys
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from kidney_disease_classification.exception import CustomException
from kidney_disease_classification.logger import logging
from kidney_disease_classification.utils.common import save_json
from kidney_disease_classification.entity.config_entity import EvaluationConfig



class Evaluation:
    def __init__(self, config: EvaluationConfig, params):
        self.config = config
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # LOAD MODEL
    # ---------------------------
    def load_model(self):
        try:
            model = models.efficientnet_b0(pretrained=False)

            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, self.params["NUM_CLASSES"])

            model.load_state_dict(
                torch.load(self.config.model_path, map_location=self.device)
            )

            model.to(self.device)
            model.eval()

            logging.info("Model loaded successfully for evaluation.")
            return model

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------------------
    # LOAD TEST DATA
    # ---------------------------
    def get_test_loader(self):
        try:
            img_size = tuple(self.params["IMAGE_SIZE"])

            test_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            test_dir = Path(self.config.dataset_dir) / "test"

            test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

            test_loader = DataLoader(
                test_dataset,
                batch_size=self.params["BATCH_SIZE"],
                shuffle=False
            )

            logging.info("Test loader created successfully.")
            return test_loader

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------------------
    # MAIN EVALUATION FUNCTION
    # ---------------------------
    def evaluate_model(self):
        try:
            logging.info("Evaluation stage started...")

            model = self.load_model()
            test_loader = self.get_test_loader()

            y_true = []
            y_pred = []
            y_probs = []

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(images).squeeze()
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).int()

                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
                    y_probs.extend(probs.cpu().numpy())

            # Metrics
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_probs)

            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()

            specificity = tn / (tn + fp)

            report = {
                "accuracy": float(acc),
                "precision": float(precision),
                "recall_sensitivity": float(recall),
                "specificity": float(specificity),
                "f1_score": float(f1),
                "roc_auc": float(auc),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn)
            }

            # Save evaluation report
            save_json(self.config.report_file, report)

            # Save confusion matrix separately
            save_json(self.config.confusion_matrix_file, {
                "confusion_matrix": cm.tolist()
            })

            logging.info("Evaluation completed successfully.")
            logging.info(f"Evaluation Report: {report}")

            return report

        except Exception as e:
            raise CustomException(e, sys)