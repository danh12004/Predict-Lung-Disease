import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from pathlib import Path
from src.cnnClassifier.entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        self.save_model(path=self.config.base_model_path, model=self.model)
    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, learning_rate):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, classes),
            nn.Softmax(dim=1)
        )

        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        return model, optimizer, criterion
    
    def update_base_model(self):
        self.full_model, self.optimizer, self.criterion = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model.state_dict(), path)