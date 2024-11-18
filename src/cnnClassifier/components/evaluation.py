import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models  
from src.cnnClassifier.entity import EvaluationConfig
from src.cnnClassifier.utils.common import save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_loader(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize (mean, std)
        ])

        dataset = datasets.ImageFolder(root=self.config.training_data, transform=transform)
        self.valid_loader = DataLoader(dataset, batch_size=self.config.params_batch_size, shuffle=False)

    @staticmethod
    def load_model(path: Path) -> torch.nn.Module:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.vgg16()
        model.load_state_dict(torch.load(path, map_location=device))  
        model.to(device)
        model.eval()
        return model

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_loader()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        criterion = torch.nn.CrossEntropyLoss()  # Adjust based on your model
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.score = {"loss": total_loss / len(self.valid_loader), "accuracy": correct / total}

    def save_score(self):
        save_json(path=Path("scores.json"), data=self.score)