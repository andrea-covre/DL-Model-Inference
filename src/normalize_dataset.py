import torch
import torchvision
import torchvision.transforms as transforms

from custom_dataset_loader import FakeCIFAR10


transform_train = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Normalize((0.4919, 0.4827, 0.4472), (0.2470, 0.2434, 0.2616)),
        #transforms.Normalize((0.4689, 0.4952, 0.4226), (0.2463, 0.2468, 0.2799)),
    ])


# train_dataset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform_train)
# valid_dataset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform_train)

train_dataset = FakeCIFAR10(train=True, transform=transform_train)
valid_dataset = FakeCIFAR10(train=False, transform=transform_train)

acc_dataset = train_dataset + valid_dataset

count = 0
pixel_sum = torch.Tensor([0, 0, 0])
sq_pixel_sum = torch.Tensor([0, 0, 0])


imgs = torch.stack([img_t for img_t, _ in acc_dataset], dim=3)
mean = imgs.view(3, -1).mean(dim=1)
std = imgs.view(3, -1).std(dim=1)

print(mean, std)

"""
Mean and std used to train resnet56 on CIFAR10: (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
Real dataset mean and std: (0.4919, 0.4827, 0.4472), (0.2470, 0.2434, 0.2616)
Fake dataset mean and std: (0.4689, 0.4952, 0.4226), (0.2463, 0.2468, 0.2799)
"""
