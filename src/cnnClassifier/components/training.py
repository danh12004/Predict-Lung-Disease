import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
from torch.utils.data import random_split
from src.cnnClassifier.entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_base_model(self):
        # Tải mô hình với strict=False để tránh lỗi thiếu key
        self.model = models.vgg16()
        self.model.load_state_dict(torch.load(self.config.updated_base_model_path, weights_only=True), strict=False)
        self.model.to(self.device)

    def train_valid_loader(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.config.params_image_size[0]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(40),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]) if self.config.params_is_augmentation else transforms.Compose([
                transforms.Resize(self.config.params_image_size[0]),
                transforms.CenterCrop(self.config.params_image_size[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(self.config.params_image_size[0]),
                transforms.CenterCrop(self.config.params_image_size[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        # Tải dữ liệu từ các thư mục đã chia sẵn
        train_dataset = datasets.ImageFolder(self.config.training_data + "/train", transform=data_transforms['train'])
        val_dataset = datasets.ImageFolder(self.config.training_data + "/val", transform=data_transforms['val'])

        # Tạo DataLoader cho các tập train và val
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.params_batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.params_batch_size, shuffle=False)

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model.state_dict(), path)

    def train(self, callback_list=None):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.config.params_learning_rate, momentum=0.9)
        
        # Training loop
        self.model.train()
        for epoch in range(self.config.params_epochs):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(self.train_loader.dataset)
            print(f'Epoch {epoch + 1}/{self.config.params_epochs}, Loss: {epoch_loss:.4f}')

            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_loss /= len(self.val_loader.dataset)
            accuracy = 100 * correct / total
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

            # Trở lại chế độ train
            self.model.train()

        # Lưu mô hình đã huấn luyện
        self.save_model(path=self.config.trained_model_path, model=self.model)