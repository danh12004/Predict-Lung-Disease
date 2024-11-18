import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model (assuming model is VGG16 or any pre-trained model)
        self.model = models.vgg16(pretrained=False)
        self.model.load_state_dict(torch.load(os.path.join("artifacts", "training", "model.pt"), map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.model.eval()

    def predict(self):
        # Define the necessary transformations for the image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

        # Load image and apply transformations
        imagename = self.filename
        test_image = Image.open(imagename)
        test_image = transform(test_image).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(test_image)
            _, predicted = torch.max(outputs, 1)
        
        # Map prediction to class label
        result = predicted.item()
        if result == 0:
            prediction = 'Lung Opacity'
        elif result == 1:
            prediction = 'Normal'
        else:
            prediction = 'Viral Pneumonia'

        return [{"image": prediction}]