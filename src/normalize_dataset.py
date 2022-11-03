import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt

transform_train = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

base_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)

count = 0
pixel_sum = torch.Tensor([0, 0, 0])
sq_pixel_sum = torch.Tensor([0, 0, 0])

for i in range(1):
    img, label = base_dataset[i]
    pixel_sum += img.sum(dim=[1, 2])
    
print(pixel_sum)