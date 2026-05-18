import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path

import mlflow
import mlflow.pytorch


from kidney_disease_classification.entity.config_entity import TrainingConfig
from kidney_disease_classification.exception import CustomException
from kidney_disease_classification.logger import logging


class Training:
    def __init__(self, config: TrainingConfig, params):
        self.config = config
        self.params = params

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # 1. DATA LOADERS (TRAIN + VAL ONLY)
    # ---------------------------
    def get_data_loaders(self):
        try:
            logging.info("Loading train and validation datasets...")

            img_size = tuple(self.params["IMAGE_SIZE"])

            # Train transform (augmentation only here)
            train_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            # Validation transform (NO augmentation)
            val_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            train_dir = Path(self.config.dataset_dir) / "train"
            val_dir = Path(self.config.dataset_dir) / "val"

            train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
            val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.params["BATCH_SIZE"],
                shuffle=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.params["BATCH_SIZE"],
                shuffle=False
            )

            logging.info("Data loaders ready")

            return train_loader, val_loader

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------------------
    # 2. MODEL BUILDING
    # ---------------------------
    def build_model(self):
        try:
            logging.info("Building EfficientNet model...")

            model = models.efficientnet_b0(pretrained=True)

            # Freeze backbone
            for param in model.features.parameters():
                param.requires_grad = False

            # Replace classifier
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, self.params["NUM_CLASSES"])

            return model.to(self.device)

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------------------
    # 3. TRAIN ONE EPOCH
    # ---------------------------
    def train_one_epoch(self, model, loader, criterion, optimizer):
        model.train()

        total_loss = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device).float()

            optimizer.zero_grad()

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    # ---------------------------
    # 4. VALIDATION (ONLY FOR MONITORING)
    # ---------------------------
    def validate(self, model, loader, criterion):
        model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float()

                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)

                total_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        return total_loss / len(loader), accuracy

    # ---------------------------
    # 5. TRAINING PIPELINE
    # ---------------------------
    def initiate_training(self):
        try:
            logging.info("Training stage started")

            train_loader, val_loader = self.get_data_loaders()
            model = self.build_model()

            criterion = nn.BCEWithLogitsLoss()

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.params["LEARNING_RATE"],
                weight_decay=self.params["WEIGHT_DECAY"]
            )

            best_val_acc = 0.0
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)  # Adjust if your MLflow server is hosted elsewhere
            mlflow.set_experiment(self.config.mlflow_experiment_name)

            with mlflow.start_run(run_name="training_run"):

                # Log hyperparameters
                mlflow.log_params({
                    "MODEL_NAME": self.params["MODEL_NAME"],
                    "PRETRAINED": self.params["PRETRAINED"],
                    "IMAGE_SIZE": self.params["IMAGE_SIZE"],
                    "BATCH_SIZE": self.params["BATCH_SIZE"],
                    "EPOCHS": self.params["EPOCHS"],
                    "LEARNING_RATE": self.params["LEARNING_RATE"],
                    "WEIGHT_DECAY": self.params["WEIGHT_DECAY"],
                    "LOSS_FUNCTION": self.params["LOSS_FUNCTION"],
                    "OPTIMIZER": self.params["OPTIMIZER"]
                })

                for epoch in range(self.params["EPOCHS"]):
                    train_loss = self.train_one_epoch(model, train_loader, criterion, optimizer)
                    val_loss, val_acc = self.validate(model, val_loader, criterion)

                    logging.info(
                        f"Epoch [{epoch+1}/{self.params['EPOCHS']}] "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_acc:.4f}"
                    )

                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    mlflow.log_metric("val_accuracy", val_acc, step=epoch)

                    # Save best model
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc

                        model_path = os.path.join(self.config.model_dir, self.config.model_name)

                        torch.save(model.state_dict(), model_path)

                        logging.info(f"Best model saved with accuracy: {best_val_acc:.4f}")

                mlflow.log_metric("best_val_accuracy", best_val_acc)
                # mlflow.log_artifact(model_path, artifact_path="model")
                mlflow.pytorch.log_model(pytorch_model=model, artifact_path="model")

                logging.info("Model artifact logged to MLflow")

            logging.info("Training stage completed successfully")
            return True

        except Exception as e:
            raise CustomException(e, sys)