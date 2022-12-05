"""
Class to train, evaluate and save a model

main() contans an example of how to use this class
"""

import os
import torch
import torchvision
import torch.optim as optim 
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix

from src.custom_dataset_loader import FakeCIFAR10
from src.mixed_dataset_2 import MixedCIFAR10_2
from src.mixed_dataset_20 import MixedCIFAR10_20
from src.resnet_mixed import get_custom_output_resent

torch.set_printoptions(linewidth=120)


class Trainer:
    def __init__(self, model, train_loader, test_loader, optimizer, criterion, epochs):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f'Using device: {self.device}')
        
    def train(self):
        self.model.to(self.device)
        self.model.train()

        min_loss = float('inf')
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                
                print(f"{i/len(self.train_loader)*100:.2f}%: {loss.item():.4f}", end='\r')

            print(f'\nEpoch: {epoch}, running loss: {running_loss}')
            
            if (running_loss < min_loss):
                min_loss = running_loss
                os.makedirs(os.path.dirname('saved_models/fake_resnet56_auto_save.pth'), exist_ok=True)
                torch.save(self.model.state_dict(), 'saved_models/fake_resnet56_auto_save.pth')
            
        print('Finished Training')

    def save_model(self, path='saved_models/fake_resnet56.pth'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            path = path.replace('.pth', '_new.pth')
            
        torch.save(self.model.state_dict(), path)
        
    def evaluate(self):
        self.model.eval()
        
        confusion_matrix = ConfusionMatrix(num_classes=10, normalize='true').to(self.device)
        
        correct_predictions = 0
        with torch.no_grad():
            for idx, data in enumerate(self.test_loader):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                
                out = self.model(images)
                predictions = torch.argmax(out, dim=1)
                
                correct_predictions += torch.sum(predictions==labels)
                confusion_matrix.update(predictions, labels)    
                
                print(f"{round(idx/len(self.test_loader)*100, 2)}% \t Correct Predictions: {correct_predictions:<20}", end='\r')

        accuracy = correct_predictions / len(self.test_loader.dataset)
                    
        return accuracy, confusion_matrix
    
    def run(self, save_model=True):
        self.train()
        
        if save_model:
            self.save_model()
            
        return self.evaluate()
    
    def _load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        
    def test_loaded_model(self, path='saved_models/fake_resnet56.pth'):
        self._load_model(path)
        return self.evaluate()

def main():

    fake_train_set = FakeCIFAR10(train=True, transform=FakeCIFAR10.standard_transform)
    fake_test_set = FakeCIFAR10(train=False, transform=FakeCIFAR10.standard_transform)
    
    mixed_2_train_set = MixedCIFAR10_2(train=True, transform=MixedCIFAR10_2.standard_transform)
    mixed_2_test_set = MixedCIFAR10_2(train=False, transform=MixedCIFAR10_2.standard_transform)
    
    real_test_set = torchvision.datasets.CIFAR10(root='./pretrained_models', train=False, download=True, transform=FakeCIFAR10.standard_transform)
    
    # Dataset selection ----------------------
    train_set = mixed_2_train_set
    test_set = mixed_2_test_set

    train_dataloader = DataLoader(train_set, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=256, shuffle=True)

    resnet56 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=False)
    repvgg_a2 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a2", pretrained=True)
    
    resent52_2 = get_custom_output_resent(20)
    
    
    # Model selection ----------------------
    model = resent52_2

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)

    trainer = Trainer(model, train_dataloader, test_dataloader, optimizer, criterion, epochs=200)
    
    # To train and save model
    accuracy, confusion_matrix = trainer.run()
    
    # To test saved model
    #accuracy, confusion_matrix = trainer.test_loaded_model("fake_resnet56_auto_save.pth")
    
    # To test model from memory
    #accuracy, confusion_matrix = trainer.evaluate()
    
    
    print(f"Accuracy: {accuracy*100}%")
    print(confusion_matrix.compute())


if __name__ == "__main__":
    main()