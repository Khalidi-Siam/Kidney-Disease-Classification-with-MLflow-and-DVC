import os
import sys
import json
from PIL import Image, ImageFile
from pathlib import Path

from kidney_disease_classification.exception import CustomException
from kidney_disease_classification.entity.config_entity import DataValidationConfig

# Allow truncated images (common in medical datasets)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            validation_status = True
            report = {}

            dataset_path = Path(self.config.dataset_dir)

            # 1. Check dataset directory exists
            if not dataset_path.exists():
                validation_status = False
                report["error"] = f"Dataset directory not found: {dataset_path}"
            else:
                class_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
                report["class_folders_found"] = [f.name for f in class_folders]

                # 2. Check at least 2 classes exist
                if len(class_folders) < 2:
                    validation_status = False
                    report["error"] = "Less than 2 class folders found (binary classification required)."

                allowed_extensions = [".jpg", ".jpeg", ".png"]

                total_images = 0
                corrupted_images = []
                non_rgb_images = []

                class_counts = {}

                # 3. Validate each class folder
                for class_folder in class_folders:
                    images = []

                    for ext in allowed_extensions:
                        images.extend(class_folder.glob(f"*{ext}"))

                    class_counts[class_folder.name] = len(images)
                    total_images += len(images)

                    # 4. Image validation
                    for img_path in images:
                        try:
                            with Image.open(img_path) as img:
                                # Step 1: verify file integrity
                                img.verify()

                            # Step 2: reopen and fully load image
                            with Image.open(img_path) as img:
                                img.load()

                                # Step 3: ensure RGB compatibility (important for CNNs)
                                if img.mode not in ["RGB", "L"]:
                                    non_rgb_images.append(str(img_path))

                        except Exception:
                            corrupted_images.append(str(img_path))

                # 5. Save report data
                report["total_images"] = total_images
                report["class_counts"] = class_counts
                report["corrupted_images_count"] = len(corrupted_images)
                report["non_rgb_images_count"] = len(non_rgb_images)
                report["corrupted_images"] = corrupted_images
                report["non_rgb_images"] = non_rgb_images

                # 6. Final validation decision
                if total_images == 0:
                    validation_status = False
                    report["error"] = "No images found in dataset."

                if len(corrupted_images) > 0:
                    validation_status = False
                    report["error"] = "Dataset contains corrupted images."

            # 7. Save report JSON
            os.makedirs(self.config.root_dir, exist_ok=True)
            with open(self.config.report_file, "w") as f:
                json.dump(report, f, indent=4)

            # 8. Save status file
            with open(self.config.status_file, "w") as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise CustomException(e, sys)