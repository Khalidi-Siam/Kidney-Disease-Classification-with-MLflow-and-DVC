from kidney_disease_classification.config.configuration import ConfigurationManager
from kidney_disease_classification.components.training import Training
from kidney_disease_classification.exception import CustomException
from kidney_disease_classification.logger import logging
import sys



STAGE_NAME = "Training stage"

class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()

            training_config = config.get_training_config()
            training = Training(config=training_config)
            training.get_base_model()
            training.train_valid_generator()
            training.train()
            
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    try:
        logging.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        training_pipeline = TrainingPipeline()
        training_pipeline.main()
        logging.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e, sys)