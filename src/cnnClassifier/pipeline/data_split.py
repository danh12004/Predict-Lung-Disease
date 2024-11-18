import os
from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components.data_split import DataSplitter
from src.cnnClassifier import logger

STAGE_NAME = "Data Split"

class DataSplitterPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_split_config = config.get_data_split_config()

        data_splitter = DataSplitter(config=data_split_config)

        source_dir = os.path.join(data_ingestion_config.unzip_dir, "Lung X-Ray Image")
        data_splitter.split_data(source_dir)