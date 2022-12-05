import torch

from torchsummary import summary

def get_custom_output_resent(output_size):
    resnet56 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=False)
    resnet56.fc = torch.nn.Linear(64, output_size)
    return resnet56
