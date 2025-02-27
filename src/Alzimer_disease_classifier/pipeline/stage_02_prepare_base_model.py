from Alzimer_disease_classifier.config.configuration import ConfigurationManager
from Alzimer_disease_classifier.components.prepare_base_model import PrepareBaseModel
from Alzimer_disease_classifier import logger
import sys
from Alzimer_disease_classifier.exception import RenelException
from Alzimer_disease_classifier.logger import logging
import os


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
            # Save the full model with custom layers
            PrepareBaseModel.save_model(
                path=prepare_base_model_config.updated_base_model_path,
                model=prepare_base_model.full_model)
        except Exception as e:
            raise RenelException(e, sys)


STAGE_NAME = "Prepare base model stage"
if __name__ == "__main__":
    try:
        logging.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<\n\nx==========x")
        preparebase_model = PrepareBaseModelTrainingPipeline()
        preparebase_model.main()
        logging.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logging.exception(e)
        raise RenelException(e, sys)
