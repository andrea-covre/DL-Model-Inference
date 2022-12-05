
import os
import torchvision
import torchvision.transforms as transforms

from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

class Item:
    def __init__(self, type, img, label):
        if type not in ["real", "fake"]:
            raise ValueError("type should be either 'real' or 'fake'")
        self.type = type
        self.img = img
        self.label = label if type == "fake" else label + 10

class MixedCIFAR10_20(Dataset):
    TRAIN_PATH = "datasets/Fake-CIFAR-10-training-data"
    VALIDATION_PATH = "datasets/Fake-CIFAR-10-validation-data"
    CIFAR10_CLASSES_MAP = {
        "airplane": 0,
        "automobile": 1, 
        "bird": 2, 
        "cat": 3, 
        "deer": 4, 
        "dog": 5, 
        "frog": 6, 
        "horse": 7, 
        "ship": 8, 
        "truck": 9
    }
    
    """
    Mean and std used to train resnet56 on CIFAR10: (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    Real dataset mean and std: (0.4919, 0.4827, 0.4472), (0.2470, 0.2434, 0.2616)
    Fake dataset mean and std: (0.4689, 0.4952, 0.4226), (0.2463, 0.2468, 0.2799)
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    standard_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    
    def __init__(self, train=True, transform=None, target_transform=None):
        self.img_dir = self.TRAIN_PATH if train else self.VALIDATION_PATH
        
        self.transform = transform
        self.target_transform = target_transform
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.classes_dir = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        
        self.classes = []
        for type in ["fake", "real"]:
            for class_name in self.classes_dir:
                self.classes.append(type + "_" + class_name)
        

        self.data = []
        
        # Adding fake CIFAR10 images
        class_index = 0
        for class_dir in self.classes_dir:
            
            if class_dir == ".DS_Store":
                continue

            for img_name in os.listdir(os.path.join(self.img_dir, class_dir)):
                
                sample = Item("fake", os.path.join(self.img_dir, class_dir, img_name), self.CIFAR10_CLASSES_MAP[class_dir])
                self.data.append(sample)
                
            class_index += 1
            
        # Adding Real CIFAR10 images
        real_cifar = torchvision.datasets.CIFAR10(root='./pretrained_models', train=train, download=True, transform=transform)
        for sample in real_cifar:
            self.data.append(Item("real", sample[0], sample[1]))
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #image = read_image(self.img_paths[idx])
        
        if self.data[idx].type == "fake":
            image = Image.open(self.data[idx].img)
            
            if self.transform:
                image = self.transform(image)
                
            if self.target_transform:
                label = self.target_transform(label)
                
        else:
            #image = self.tensor_transform(self.data[idx].img)
            image = self.data[idx].img
            
        
        label = self.data[idx].label

            
        return image, label
    
def main():
    dataset = MixedCIFAR10_2(train=True)
    print(">>>> dataset")

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(">>>> loader")

    for i in range(0, 5):
        #image, label = dataset[i]
        image, label = next(iter(train_dataloader))
        image = image[0, :, :, :]
        print(dataset.classes[label])
        plt.imshow(image.permute(1, 2, 0))
        plt.show()
        
if __name__ == "__main__":
    main()
