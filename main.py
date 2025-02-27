from Alzimer_disease_classifier.logger import logging
from Alzimer_disease_classifier.exception import RenelException
import sys
from Alzimer_disease_classifier.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from Alzimer_disease_classifier.pipeline.stage_02_prepare_base_model import (
    PrepareBaseModelTrainingPipeline,
)
from Alzimer_disease_classifier.pipeline.stage_03_model_training import (
    ModelTrainingPipeline,
)

STAGE_NAME = "Data Ingestion Stage"
try:
    logging.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<\n\nx==========x")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logging.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logging.exception(e)
    raise RenelException(e, sys)


STAGE_NAME = "Prepare base model stage"
try:

    logging.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<\n\nx==========x")
    preparebase_model = PrepareBaseModelTrainingPipeline()
    preparebase_model.main()
    logging.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logging.exception(e)
    raise RenelException(e, sys)


STAGE_NAME = "Model Training"
try:

    logging.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<\n\nx==========x")
    model_trainer_pipeline = ModelTrainingPipeline()
    model_trainer_pipeline.main()
    logging.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    raise RenelException(e, sys)


# STAGE_NAME = "Model Evaluation"
# try:

#     logging.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<\n\nx==========x")

#     evaluation = EvaluationPipeline()
#     evaluation.main()
#     logging.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

# except Exception as e:
#     raise RenelException(e, sys)
