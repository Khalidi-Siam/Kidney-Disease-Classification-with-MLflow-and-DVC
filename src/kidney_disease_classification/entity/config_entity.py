from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    dataset_dir: Path
    status_file: Path
    report_file: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    dataset_dir: Path
    status_file: Path
    transformed_dir: Path


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    dataset_dir: Path
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    model_dir: Path
    model_name: str
    save_best_only: bool
    monitor_metric: str
    early_stopping: bool
    patience: int


@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    dataset_dir: Path
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    model_path: Path
    report_file: Path
    confusion_matrix_file: Path


@dataclass(frozen=True)
class XAIConfig:
    root_dir: Path
    dataset_dir: Path
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    model_path: Path
    output_dir: Path
    num_samples: int