import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dataset import BrainDataset
# from model import AlzheimerDetectionModel 
from tqdm import tqdm
import wandb
from resnet_model import ResNetClassifier
from torch.cuda.amp import GradScaler, autocast  # Import AMP modules


def validate_model(model, val_dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for val_data, val_labels in val_dataloader:
            val_outputs = model(val_data.to(device))
            val_loss += criterion(val_outputs, val_labels.to(device)).item()
            _, predicted = torch.max(val_outputs, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(val_labels.cpu().tolist())
            total += val_labels.size(0)
            correct += (predicted == val_labels.to(device)).sum().item()
    
    val_accuracy = correct / total
    return val_loss, val_accuracy

def calculate_accuracy(predicted, labels):
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct, total

def log_metrics(epoch, step, loss, val_loss, val_accuracy, train_accuracy=None):
    wandb.log({'epoch': epoch+1, 'loss': loss})
    if val_loss is not None and val_accuracy is not None:
        wandb.log({'val_loss': val_loss, 'val_accuracy': val_accuracy})
    if train_accuracy is not None:
        wandb.log({'train_accuracy': train_accuracy})

def train_model(data_dir, device):
    bsz = 4
    lr = 0.001 
    train_data = BrainDataset(data_dir, 'train')
    val_data = BrainDataset(data_dir, 'val')

    train_dataloader = DataLoader(train_data, batch_size=bsz, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=bsz, shuffle=False)

    model = ResNetClassifier(
        layers=[3, 8, 36, 3], 
        sample_input_D=32, 
        sample_input_H=32, 
        sample_input_W=32, 
        num_seg_classes=2, 
        shortcut_type='B'
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    epochs = 20

    scaler = GradScaler()

    wandb.login(key="39e65ab86c39c92f1b18458c6cc56fee691e0705")
    wandb.init(project='ADNI-Detection', config={'lr': 0.001, 'batch_size': 8})

    print("starting to train")
    # Training loop
    for epoch in range(epochs):
        correct_train = 0
        total_train = 0
        for i, (data, labels) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch+1}')):

            optimizer.zero_grad()
            with autocast():  # Enable mixed precision for training
                outputs = model(data.to(device))
                loss = criterion(outputs, labels.to(device))
            scaler.scale(loss).backward()  # Scale the loss and call backward
            scaler.step(optimizer)  # Step the optimizer
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.update()  # Update the scaler

            _, predicted = torch.max(outputs, 1)
            correct, total = calculate_accuracy(predicted, labels.to(device))
            correct_train += correct
            total_train += total
            
            # Log loss and training accuracy for the batch at every step
            train_accuracy = correct / total
            log_metrics(epoch, i+1, loss.item(), None, None, train_accuracy)
            # val_loss, val_accuracy = validate_model(model, val_dataloader, criterion, device)
            # Perform validation 3 times in an epoch
            if (i+1) % (len(train_dataloader) // 2) == 0:
                val_loss, val_accuracy = validate_model(model, val_dataloader, criterion, device)
                log_metrics(epoch, i+1, loss.item(), val_loss/len(val_dataloader), val_accuracy)

        # Final validation at the end of the epoch
        # val_loss, val_accuracy = validate_model(model, val_dataloader, criterion, device)
        train_accuracy = correct_train / total_train
        log_metrics(epoch, 'end', loss.item(), val_loss/len(val_dataloader), val_accuracy, train_accuracy)
        print(f'Epoch {epoch+1}, Loss: {loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Training Accuracy: {train_accuracy}')

        torch.save(model.state_dict(), 'model_path.pth')
        model.load_state_dict(torch.load('model_path.pth'))

    wandb.save('model_path.pth')

if __name__ == "__main__":
    # Call the function to train the model
    data_dir = "/data2/om/ADNI dataset/data"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(data_dir, device=device)
