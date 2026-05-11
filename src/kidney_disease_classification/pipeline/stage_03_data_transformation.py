import sys
from kidney_disease_classification.exception import CustomException
from kidney_disease_classification.config.configuration import ConfigurationManager
from kidney_disease_classification.components.data_transformation import DataTransformation
from kidney_disease_classification.logger import logging


STAGE_NAME = "Data Transformation stage"


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            data_transformation_config = config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config, params=config_manager.params)
            data_transformation.initiate_transformation()
        except Exception as e:
            raise CustomException(e, sys)



if __name__ == "__main__":
    try:
        logging.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        prepare_base_model = DataTransformationTrainingPipeline()
        prepare_base_model.main()
        logging.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e, sys)