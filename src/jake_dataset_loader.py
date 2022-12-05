
import os
import torchvision.transforms as transforms

from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

class FakeCIFAR10(Dataset):
    TRAIN_PATH = "../jake_images/raw_fake_dataset"
    VALIDATION_PATH = "../jake_images/raw_fake_dataset"
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
    
    def __init__(self, train=True, transform=standard_transform, target_transform=None):
        self.img_dir = self.TRAIN_PATH if train else self.VALIDATION_PATH
        self.transform = transform
        self.target_transform = target_transform
        
        self.classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        self.img_labels = []
        self.img_paths = []
        
        class_index = 0
        for class_dir in self.classes:
            
            if class_dir == ".DS_Store":
                continue

            for img_name in os.listdir(os.path.join(self.img_dir, class_dir)):
                if img_name == ".DS_Store":
                    continue
                self.img_paths.append(os.path.join(self.img_dir, class_dir, img_name))
                self.img_labels.append(self.CIFAR10_CLASSES_MAP[class_dir])
                
            class_index += 1

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #image = read_image(self.img_paths[idx])
        image = Image.open(self.img_paths[idx])
        label = self.img_labels[idx]
        
        if self.transform:
            #print("transformed!")
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label
    
def main():
    dataset = FakeCIFAR10(train=True)

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i in range(0, 5):
        #image, label = dataset[i]
        image, label = next(iter(train_dataloader))
        image = image[0, :, :, :]
        print(dataset.label_names[label])
        plt.imshow(image.permute(1, 2, 0))
        plt.show()
        
if __name__ == "__main__":
    main()
