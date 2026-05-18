import sys
from kidney_disease_classification.config.configuration import ConfigurationManager
from kidney_disease_classification.components.prediction import Prediction
from kidney_disease_classification.exception import CustomException
from kidney_disease_classification.logger import logging


STAGE_NAME = "Prediction stage"

class PredictionPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.prediction_config = config.get_prediction_config()
        self.params = config.params
        self.prediction = Prediction(config=self.prediction_config, params=self.params)

    def main(self, img_path):
        try:
            prediction_report = self.prediction.predict(img_path)
            return prediction_report
        
        except Exception as e:
            raise CustomException(e, sys)
    