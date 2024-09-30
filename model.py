import torch
from torchvision import models, transforms
import requests

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = requests.get(LABELS_URL)
labels = response.json() 
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_model():
    return model, labels, preprocess
