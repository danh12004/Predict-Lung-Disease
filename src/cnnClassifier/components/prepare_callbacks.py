import os
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from src.cnnClassifier.entity import PrepareCallbacksConfig

class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config
    
    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}"
        )

        self.writer = SummaryWriter(log_dir=tb_running_log_dir)
        return self.writer
    
    def _create_ckpt_callbacks(self, model, optimizer, epoch):
        checkpoint_path = self.config.checkpoint_model_filepath

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': self.config.loss 
        }, checkpoint_path)

    def get_tb_ckpt_callbacks(self, model=None, optimizer=None, epoch=None):
        return {
            "tensorboard": self._create_tb_callbacks,
            "checkpoint": lambda: self._create_ckpt_callbacks(model, optimizer, epoch)
        }