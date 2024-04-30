import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class AlzheimerDetectionModel(nn.Module):
    def __init__(self):
        super(AlzheimerDetectionModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 10 * 15 * 10, 1024),  # Adjust the input size
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            # nn.BatchNorm1d(1024)
        )
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.fc1(x)
        x = nn.functional.leaky_relu(self.fc2(x))
        x = nn.functional.dropout(x, 0.6)
        x = nn.functional.leaky_relu(self.fc3(x))
        x = nn.functional.dropout(x, 0.7)
        x = self.fc4(x)
        return self.softmax(x)

if __name__ == "__main__":
    from dataset import BrainDataset
    braindataset = BrainDataset(data_dir="/data2/om/ADNI dataset/data")
    dataloader = DataLoader(braindataset, batch_size=8, shuffle=True)
    model = AlzheimerDetectionModel()
    for data, labels in dataloader:
        print(data.shape)
        break
    # data = braindataset[0][0].unsqueeze(0).unsqueeze(0)
    out = model(data)
    # print(out)
    print(data.shape, out.shape)