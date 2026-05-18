import sys
from kidney_disease_classification.config.configuration import ConfigurationManager
from kidney_disease_classification.components.xai import XAI
from kidney_disease_classification.exception import CustomException
from kidney_disease_classification.logger import logging


STAGE_NAME = "XAI stage"

class XAIPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            xai_config = config.get_xai_config()
            xai = XAI(config=xai_config, params=config.params)
            xai.generate_gradcam_outputs()
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    try:
        logging.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        xai_pipeline = XAIPipeline()
        xai_pipeline.main()
        logging.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e, sys)