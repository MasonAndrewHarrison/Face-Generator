import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder




transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])




dataset = ImageFolder(root='dataset/', transform=transform)

image, label = dataset[3543]
image = image.permute(1, 2, 0)
image = image * 0.5 + 0.5

print(image.shape)
plt.imshow(image)
plt.show()




