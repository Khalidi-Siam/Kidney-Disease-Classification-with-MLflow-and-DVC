from kidney_disease_classification.exception import CustomException
import sys
import os
from kidney_disease_classification.entity.config_entity import DataValitationConfig

class DataValidation:
    def __init__(self, config: DataValitationConfig):
        self.config = config

    def validate_all_files_exist(self)-> bool:
        try:
            validation_status = None

            all_files = os.listdir(self.config.unzip_data_dir)

            if len(all_files) == 0:
                validation_status = False
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation status: {validation_status}")
            else:
                validation_status = True
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise CustomException(e, sys)