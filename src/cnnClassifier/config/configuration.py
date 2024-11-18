import os
from src.cnnClassifier.constants import *
from src.cnnClassifier.utils.common import read_yaml, create_directories
from src.cnnClassifier.entity import (DataIngestionConfig, DataSplitConfig, PrepareBaseModelConfig, 
                                      PrepareCallbacksConfig, TrainingConfig, EvaluationConfig)

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params= read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    
    def get_data_split_config(self) -> DataSplitConfig:
        config = self.config.data_split
        create_directories([config.root_dir, config.train_dir, config.val_dir, config.test_dir])

        data_split_config = DataSplitConfig(
            root_dir=config.root_dir,
            train_dir=config.train_dir,
            val_dir=config.val_dir,
            test_dir=config.test_dir,
            split_ratios=tuple(config.split_ratios)
        )
        return data_split_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=config.root_dir,
            base_model_path=config.base_model_path,
            updated_base_model_path=config.updated_base_model_path,
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    
    def get_prepare_callbacks_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        prepare_callbacks_config = PrepareCallbacksConfig(
            root_dir=config.root_dir,
            tensorboard_root_log_dir=config.tensorboard_root_log_dir,
            checkpoint_model_filepath=config.checkpoint_model_filepath
        )

        return prepare_callbacks_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_split.root_dir)
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=training.root_dir,
            trained_model_path=training.trained_model_path,
            updated_base_model_path=prepare_base_model.updated_base_model_path,
            training_data=training_data,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE
        )

        return training_config
    
    def get_validation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.pt",
            training_data="artifacts/data_split/test",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        
        return eval_config