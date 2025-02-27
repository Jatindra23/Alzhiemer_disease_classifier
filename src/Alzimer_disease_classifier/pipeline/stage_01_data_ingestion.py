from Alzimer_disease_classifier.config.configuration import ConfigurationManager
from Alzimer_disease_classifier.components.data_ingestion import DataIngestion
from Alzimer_disease_classifier.exception import RenelException
from Alzimer_disease_classifier.logger import logging
from pathlib import Path


import sys


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:

            logging.info("Entered in ingestion pipline main funcion")
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_path = Path(data_ingestion_config.data_path)
            data_ingestion.get_data_from_folder(data_path=data_path)
            logging.info("exited from ingestion pipline main funcion")

        except Exception as e:
            raise RenelException(e, sys)


if __name__ == "__main__":
    try:
        logging.info(f">>>>>>> stage Data Ingestion started <<<<<<\n\nx==========x")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logging.info(f">>>>>>> stage Data Ingestion completed <<<<<<\n\nx==========x")

    except Exception as e:
        logging.exception(e)
        raise RenelException(e, sys)
