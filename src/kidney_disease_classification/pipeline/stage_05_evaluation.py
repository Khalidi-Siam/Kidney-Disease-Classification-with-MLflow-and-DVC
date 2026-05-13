import sys
from kidney_disease_classification.config.configuration import ConfigurationManager
from kidney_disease_classification.components.evaluation import Evaluation
from kidney_disease_classification.exception import CustomException
from kidney_disease_classification.logger import logging


STAGE_NAME = "Evaluation stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            evaluation_config = config.get_evaluation_config()
            evaluation = Evaluation(config=evaluation_config, params=config.params)
            evaluation_report = evaluation.evaluate_model()
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    try:
        logging.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        training_pipeline = EvaluationPipeline()
        training_pipeline.main()
        logging.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e, sys)