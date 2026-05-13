import os
import random
import sys
import shutil
from kidney_disease_classification.exception import CustomException
from kidney_disease_classification.logger import logging
from kidney_disease_classification.entity.config_entity import DataTransformationConfig
from pathlib import Path


class DataTransformation:
    def __init__(self, config: DataTransformationConfig, params):
        self.config = config
        self.params = params

        self.img_size = tuple(self.params["IMAGE_SIZE"])
        self.train_split = self.params["TRAIN_SPLIT"]
        self.val_split = self.params["VAL_SPLIT"]
        self.test_split = self.params["TEST_SPLIT"]
        self.seed = self.params["SEED"]

        random.seed(self.seed)

    # ---------------------------
    # 1. CHECK VALIDATION STATUS
    # ---------------------------
    def check_validation_status(self):
        try:
            if not os.path.exists(self.config.status_file):
                logging.error("Validation status file not found.")
                return False

            with open(self.config.status_file, "r") as f:
                status_text = f.read().strip()

            if "True" in status_text:
                logging.info("Validation passed. Starting transformation.")
                return True
            else:
                logging.error("Validation failed. Stopping data transformation.")
                return False

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------------------
    # 2. LOAD ALL IMAGE PATHS
    # ---------------------------
    def get_image_paths(self):
        dataset_path = Path(self.config.dataset_dir)

        class_folders = [f for f in dataset_path.iterdir() if f.is_dir()]

        data = []

        for class_folder in class_folders:
            label = class_folder.name

            for img_path in class_folder.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    data.append((str(img_path), label))

        return data

    # ---------------------------
    # 3. SPLIT DATASET
    # ---------------------------
    def split_data(self, data):
        random.shuffle(data)

        total = len(data)
        train_end = int(total * self.train_split)
        val_end = train_end + int(total * self.val_split)

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        return train_data, val_data, test_data

    # ---------------------------
    # 4. SAVE SPLIT DATA
    # ---------------------------
    def save_split(self, dataset, split_name):
        for img_path, label in dataset:
            src = Path(img_path)

            dst_dir = Path(self.config.transformed_dir) / split_name / label
            dst_dir.mkdir(parents=True, exist_ok=True)

            dst_path = dst_dir / src.name

            shutil.copy2(src, dst_path)
    # ---------------------------
    # 6. MAIN METHOD
    # ---------------------------
    def initiate_transformation(self):
        try:
            logging.info("Starting data transformation stage")

            # Step 1: check validation
            if not self.check_validation_status():
                logging.error("Data validation failed. Transformation aborted.")
                return False

            # Step 2: load dataset
            data = self.get_image_paths()
            logging.info(f"Total images found: {len(data)}")

            # Step 3: split
            train_data, val_data, test_data = self.split_data(data)

            logging.info(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

            # Step 4: save split dataset
            self.save_split(train_data, "train")
            self.save_split(val_data, "val")
            self.save_split(test_data, "test")

            # Step 5: save status
            os.makedirs(self.config.root_dir, exist_ok=True)
            with open(self.config.status_file, "w") as f:
                f.write("True")

            logging.info("Data transformation completed successfully")

            return True

        except Exception as e:
            raise CustomException(e, sys)