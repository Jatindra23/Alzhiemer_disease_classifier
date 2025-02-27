from Alzimer_disease_classifier.config.configuration import ConfigurationManager
from Alzimer_disease_classifier.components.model_evaluation_mlflow import Evaluation
from Alzimer_disease_classifier.exception import RenelException
from Alzimer_disease_classifier.logger import logging
import sys


STAGE_NAME = "Model Evaluation Stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            val_config = config.get_evaluation_config()
            evaluation = Evaluation(val_config)
            evaluation.evaluation()
            evaluation.save_score()
            # evaluation.log_into_mlflow()

        except Exception as e:
            raise RenelException(e, sys)


if __name__ == "__main__":
    try:
        obj = EvaluationPipeline()
        obj.main()
    except Exception as e:
        raise RenelException(e, sys)
