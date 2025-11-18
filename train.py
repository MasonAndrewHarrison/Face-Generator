import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib as plt
from torchvision.datasets import ImageFolder




transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])




dataset = ImageFolder(root='dataset/', transform=transform)

print(dataset[0])


