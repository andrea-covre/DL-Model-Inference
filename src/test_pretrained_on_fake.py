import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from custom_dataset_loader import FakeCIFAR10


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset = FakeCIFAR10("datasets/Fake-CIFAR-10-validation-data", transform=transform_train)
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

resnet56 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)

resnet56.eval()

for i in range(10):
    img, label = next(iter(train_dataloader))
    prediction = torch.argmax(resnet56(img))
    print(f"Prediction: {dataset.label_names[prediction]:<20}  Label: {dataset.label_names[label]:<20} >> {bool(prediction==label)}")