import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from custom_dataset_loader import FakeCIFAR10
from utils import show


transform_train = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # Original
    #transforms.Normalize((0.4919, 0.4827, 0.4472), (0.2470, 0.2434, 0.2616)) 
])

real_dataset = torchvision.datasets.CIFAR10(root='./pretrained_models', train=False, download=True, transform=transform_train)
fake_dataset = FakeCIFAR10("datasets/Fake-CIFAR-10-validation-data", transform=transform_train)

dataset = fake_dataset
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

resnet56 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)

resnet56.eval()

permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
permutations = [permutations[0]]

for permutation in permutations:
    cnt = 0
    
    iterations = 10
    for i in range(iterations):
        img, label = next(iter(train_dataloader))
        
        img = img[:, permutation]
    
        prediction = torch.argmax(resnet56(img))
        
        #print(f"Prediction: {dataset.classes[prediction]:<10}  Label: {dataset.classes[label]:<10} >> {bool(prediction==label)}")
        
        if prediction==label:
            cnt += 1
        
        img = torch.squeeze(img)
        
        show(img)
        
        if 0:
            img = torch.squeeze(img)
            r_img = torch.clone(img)
            g_img = torch.clone(img)
            b_img = torch.clone(img)
            
            r_img[[1, 2], :, :] = 0
            g_img[[0, 2], :, :] = 0
            b_img[[0, 1], :, :] = 0

            # show(r_img)
            # show(g_img)
            # show(b_img)
            
            # show(img)
            
            f, axarr = plt.subplots(2,2)
            axarr[0,0].imshow(img.squeeze().permute(1, 2, 0))
            axarr[0,1].imshow(r_img.squeeze().permute(1, 2, 0))
            axarr[1,0].imshow(g_img.squeeze().permute(1, 2, 0))
            axarr[1,1].imshow(b_img.squeeze().permute(1, 2, 0))
            plt.show()
            
        #print()
        
    print(f"Accuracy: {cnt/iterations*100}% for image permutation {permutation}")
        