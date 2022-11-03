
import os
import torchvision.transforms as transforms

from PIL import Image
from matplotlib import pyplot as plt
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class FakeCIFAR10(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
        self.label_names = []
        self.img_labels = []
        self.img_paths = []
        
        class_index = 0
        for class_dir in os.listdir(img_dir):
            if class_dir == ".DS_Store":
                continue
            
            self.label_names.append(class_dir)

            for img_name in os.listdir(os.path.join(img_dir, class_dir)):
                self.img_paths.append(os.path.join(self.img_dir, class_dir, img_name))
                self.img_labels.append(class_index)
                
            class_index += 1

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #image = read_image(self.img_paths[idx])
        image = Image.open(self.img_paths[idx])
        label = self.img_labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label
    
def main():
    dataset = FakeCIFAR10("datasets/Fake-CIFAR-10-training-data")

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i in range(0, 5):
        #image, label = dataset[i]
        image, label = next(iter(train_dataloader))
        image = image[0, :, :, :]
        print(dataset.label_names[label])
        plt.imshow(image.permute(1, 2, 0))
        plt.show()
        
if __name__ == "main":
    main()
