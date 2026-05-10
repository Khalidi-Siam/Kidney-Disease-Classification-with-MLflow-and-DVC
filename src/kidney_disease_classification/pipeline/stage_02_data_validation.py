import sys
from kidney_disease_classification.exception import CustomException
from kidney_disease_classification.config.configuration import ConfigurationManager
from kidney_disease_classification.components.data_validation import DataValidation
from kidney_disease_classification.logger import logging

STAGE_NAME = "Data Validation stage"


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            data_validation.validate_all_files_exist()
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    try:
        logging.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        data_validation = DataValidationTrainingPipeline()
        data_validation.main()
        logging.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e, sys)
