from src.cnnClassifier import logger
from src.cnnClassifier.pipeline.data_ingestion import DataIngestionTrainingPipeline
from src.cnnClassifier.pipeline.data_split import DataSplitterPipeline
from src.cnnClassifier.pipeline.prepare_base_model import PrepareBaseModelTrainingPipeline
from src.cnnClassifier.pipeline.training import TrainingPipeline
from src.cnnClassifier.pipeline.evaluation import EvaluationPipeline

STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} start <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Splitter stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} start <<<<<<")
    data_ingestion = DataSplitterPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare Base Model"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} start <<<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e


# STAGE_NAME = "Training"
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} start <<<<<<")
#     training = TrainingPipeline()
#     training.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
# except Exception as e:
#     logger.exception(e)
#     raise e


STAGE_NAME = "Evaluation"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} start <<<<<<")
    prepare_base_model = EvaluationPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e