from kidney_disease_classification.constants import *
from kidney_disease_classification.utils.common import read_yaml, create_directories
from kidney_disease_classification.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, TrainingConfig, EvaluationConfig, XAIConfig, PredictionConfig


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
            root_dir = config["root_dir"],
            dataset_dir = config["dataset_dir"],
            status_file = config["status_file"],
            transformed_dir = config["transformed_dir"]
        )

        return data_transformation_config
            

    def get_training_config(self) -> TrainingConfig:
        config = self.config["training"]

        create_directories([config["root_dir"], config["model_dir"]])

        training_config = TrainingConfig(
            root_dir=config["root_dir"],
            dataset_dir=config["dataset_dir"],
            mlflow_tracking_uri=config["mlflow_tracking_uri"],
            mlflow_experiment_name=config["mlflow_experiment_name"],
            model_dir=config["model_dir"],
            model_name=config["model_name"],
            save_best_only=config["save_best_only"],
            monitor_metric=config["monitor_metric"],
            early_stopping=config["early_stopping"],
            patience=config["patience"]
        )
        return training_config
    

    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config["evaluation"]

        create_directories([config["root_dir"]])

        evaluation_config = EvaluationConfig(
            root_dir = config["root_dir"],
            dataset_dir = config["dataset_dir"],
            mlflow_tracking_uri=config["mlflow_tracking_uri"],
            mlflow_experiment_name=config["mlflow_experiment_name"],
            model_path = config["model_path"],
            report_file = config["report_file"],
            confusion_matrix_file = config["confusion_matrix_file"]
        )

        return evaluation_config
      


    def get_xai_config(self) -> XAIConfig:
        config = self.config["xai"]

        create_directories([config["root_dir"], config["output_dir"]])

        evaluation_config = XAIConfig(
            root_dir=config["root_dir"],
            dataset_dir=config["dataset_dir"],
            mlflow_tracking_uri=config["mlflow_tracking_uri"],
            mlflow_experiment_name=config["mlflow_experiment_name"],
            model_path=config["model_path"],
            output_dir=config["output_dir"],
            num_samples=config["num_samples"]
        )
        return evaluation_config
    

    def get_prediction_config(self) -> PredictionConfig:
        config = self.config["prediction"]

        prediction_config = PredictionConfig(
            model_path=config["model_path"]
        )

        return prediction_config