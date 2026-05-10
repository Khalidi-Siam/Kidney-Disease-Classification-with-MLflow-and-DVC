import sys
from kidney_disease_classification.exception import CustomException
from kidney_disease_classification.config.configuration import ConfigurationManager
from kidney_disease_classification.components.prepare_base_model import PrepareBaseModel
from kidney_disease_classification.logger import logging

STAGE_NAME = "Prepare Base Model stage"



class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            prepare_base_model_config = config.get_prepare_base_model_config()
            prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
            prepare_base_model.get_base_model()
            prepare_base_model.update_base_model()
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    try:
        logging.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        prepare_base_model = PrepareBaseModelTrainingPipeline()
        prepare_base_model.main()
        logging.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e, sys)