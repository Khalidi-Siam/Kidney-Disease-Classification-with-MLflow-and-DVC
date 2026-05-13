from kidney_disease_classification.constants import *
from kidney_disease_classification.utils.common import read_yaml, create_directories
from kidney_disease_classification.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, TrainingConfig
import os


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config["artifacts_root"]])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]

        create_directories([config["root_dir"]])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config["root_dir"],
            source_URL=config["source_URL"],
            local_data_file=config["local_data_file"],
            unzip_dir=config["unzip_dir"] 
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config["data_validation"]

        create_directories([config["root_dir"]])

        data_validation_config = DataValidationConfig(
            root_dir=config["root_dir"],
            dataset_dir=config["dataset_dir"],
            status_file=config["status_file"],
            report_file=config["report_file"]
        )

        return data_validation_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config["data_transformation"]

        create_directories([config["root_dir"]])

        data_transformation_config = DataTransformationConfig(
            root_dir = Path(config["root_dir"]),
            dataset_dir = Path(config["dataset_dir"]),
            status_file = Path(config["status_file"]),
            transformed_dir = Path(config["transformed_dir"])
        )

        return data_transformation_config
        

    def get_training_config(self) -> TrainingConfig:
        config = self.config["training"]
        self.params = self.params
        prepare_base_model = self.config["prepare_base_model"]

        training_data = os.path.join(
            self.config["data_ingestion"]["unzip_dir"],
            "kidney-ct-scan-image" 
        )

        create_directories([Path(config["root_dir"])])

        training_config = TrainingConfig(
            root_dir=Path(config["root_dir"]),
            trained_model_path=Path(config["trained_model_path"]),
            updated_base_model_path=Path(prepare_base_model["updated_base_model_path"]),
            training_data=Path(training_data),
            params_epochs=self.params["EPOCHS"],
            params_batch_size=self.params["BATCH_SIZE"],
            params_is_augmentation=self.params["AUGMENTATION"],
            params_image_size=self.params["IMAGE_SIZE"]
        )
        return training_config
    

    def get_training_config(self) -> TrainingConfig:
        config = self.config["training"]

        create_directories([Path(config["root_dir"])])

        training_config = TrainingConfig(
            root_dir=Path(config["root_dir"]),
            dataset_dir=Path(config["dataset_dir"]),
            model_dir=Path(config["model_dir"]),
            model_name=config["model_name"],
            save_best_only=config["save_best_only"],
            monitor_metric=config["monitor_metric"],
            early_stopping=config["early_stopping"],
            patience=config["patience"]
        )
        return training_config
      