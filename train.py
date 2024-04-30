import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dataset import BrainDataset
from model import AlzheimerDetectionModel 
from tqdm import tqdm
import wandb


def validate_model(model, val_dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for val_data, val_labels in val_dataloader:
            val_outputs = model(val_data.to(device))
            val_loss += criterion(val_outputs, val_labels.to(device)).item()
            _, predicted = torch.max(val_outputs, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
    
    val_accuracy = correct / total
    return val_loss, val_accuracy

def train_model(data_dir, device):
    train_data = BrainDataset(data_dir, 'train')
    val_data = BrainDataset(data_dir, 'val')

    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=8, shuffle=False)

    model = AlzheimerDetectionModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    wandb.login(key="39e65ab86c39c92f1b18458c6cc56fee691e0705")


    wandb.init(project='ADNI-Detection', config={'lr': 0.001, 'batch_size': 8})


    print("starting to train")
    # Training loop
    for epoch in range(20):
        for data, labels in tqdm(train_dataloader, desc=f'Epoch {epoch+1}'):
            optimizer.zero_grad()
            outputs = model(data.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
        
        # Validation
        val_loss, val_accuracy = validate_model(model, val_dataloader, criterion, device)
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss/len(val_dataloader)}, Validation Accuracy: {val_accuracy}')

        wandb.log({'epoch': epoch+1, 'loss': loss.item(), 'val_loss': val_loss/len(val_dataloader), 'val_accuracy': val_accuracy})

        torch.save(model.state_dict(), 'model_path.pth')
        model.load_state_dict(torch.load('model_path.pth'))

    wandb.save('model_path.pth')

# Call the function to train the model
data_dir = "/data2/om/ADNI dataset/data"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_model(data_dir, device='cpu')
