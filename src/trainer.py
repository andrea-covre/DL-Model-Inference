"""
Class to train, evaluate and save a model

main() contans an example of how to use this class
"""

import os
import torch
import torch.optim as optim 
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix

from src.custom_dataset_loader import FakeCIFAR10


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

            print(f'Epoch: {epoch}, running loss: {running_loss}')
            torch.save(self.model.state_dict(), 'saved_models/fake_resnet56_auto_save.pth')
            
        print('Finished Training')

    def save_model(self, path='saved_models/fake_resnet56.pth'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            path = path.replace('.pth', '_new.pth')
            
        torch.save(self.model.state_dict(), path)
        
    def evaluate(self):
        self.model.eval()
        
        confusion_matrix = ConfusionMatrix(num_classes=10, normalize=None).to(self.device)
        
        correct_predictions = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                
                out = self.model(images)
                predictions = torch.argmax(out, dim=1)
                
                correct_predictions += torch.sum(predictions==labels)
                confusion_matrix.update(predictions, labels)    

        accuracy = correct_predictions / len(self.test_loader.dataset)
                    
        return accuracy, confusion_matrix
    
    def run(self, save_model=True):
        self.train()
        
        if save_model:
            self.save_model()
            
        return self.evaluate()
        

def main():

    fake_train_set = FakeCIFAR10(train=True, transform=FakeCIFAR10.standard_transform)
    fake_test_set = FakeCIFAR10(train=False, transform=FakeCIFAR10.standard_transform)

    train_dataloader = DataLoader(fake_train_set, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(fake_test_set, batch_size=256, shuffle=True)

    resnet56 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=False)
    
    model = resnet56

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)

    trainer = Trainer(model, train_dataloader, test_dataloader, optimizer, criterion, epochs=200)
    
    accuracy, confusion_matrix = trainer.run()
    print(f"Accuracy: {accuracy*100}%")
    print(confusion_matrix.compute())


if __name__ == "__main__":
    main()