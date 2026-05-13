import sys
from kidney_disease_classification.logger import logging
from kidney_disease_classification.exception import CustomException
from kidney_disease_classification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from kidney_disease_classification.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from kidney_disease_classification.pipeline.stage_04_training import TrainingPipeline
from kidney_disease_classification.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from kidney_disease_classification.pipeline.stage_05_evaluation import EvaluationPipeline



STAGE_NAME="Data Ingestion Stage"
try:
    logging.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logging.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
except Exception as e:
    raise CustomException(e, sys)


STAGE_NAME="Data Validation Stage"
try:
    logging.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logging.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
except Exception as e:
    raise CustomException(e, sys)


STAGE_NAME="Data Transformation Stage"
try:
    logging.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    prepare_base_model = DataTransformationTrainingPipeline()
    prepare_base_model.main()
    logging.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
except Exception as e:
    raise CustomException(e, sys)


STAGE_NAME = "Traning stage"
try:
    logging.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    training_pipeline = TrainingPipeline()
    training_pipeline.main()
    logging.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
except Exception as e:
    raise CustomException(e, sys)

STAGE_NAME = "Evaluation stage"
try:
    logging.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    training_pipeline = EvaluationPipeline()
    training_pipeline.main()
    logging.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
except Exception as e:
    raise CustomException(e, sys)