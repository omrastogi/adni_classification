import torch
import torch.nn as nn
from resnet import ResNet, Bottleneck


class ResNetClassifier(nn.Module):
    def __init__(self, layers, num_classes=2, **kwargs):
        super(ResNetClassifier, self).__init__()
        # Initialize the ResNet model with the provided block and layers
        self.resnet = ResNet(Bottleneck, layers, **kwargs)
        # Assuming the output of the last ResNet block is 512 * expansion factor
        # You need to confirm the expansion factor from your block, typically 4 for Bottleneck
        pool_sz = (20, 20, 20)
        self.avgpool = nn.AdaptiveAvgPool3d(pool_sz)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(pool_sz[0] * pool_sz[1] * pool_sz[2] * num_classes, 2048)
        self.fc2 = nn.Linear(2048, 16)
        self.fc3 = nn.Linear(16, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through the initial ResNet model
        # [B, 1, 170, 170, 170]
        x = self.resnet(x)
        if torch.isnan(x).any():
            print("NaN detected after ResNet")
            return x
        # Global average pooling and fully connected layer for classification
        # [B, 2, 44, 44, 44]
        x = self.avgpool(x)
        if torch.isnan(x).any():
            print("NaN detected after avgpool")
            return x
        # [B, 2, 8, 8, 8]
        x = torch.flatten(x, 1)
        if torch.isnan(x).any():
            print("NaN detected after flatten")
            return x

        x = self.dropout(x)
        x = self.fc1(x)
        if torch.isnan(x).any():
            print("NaN detected after fc1")
            return x

        x = self.dropout(x)
        x = self.fc2(x)
        if torch.isnan(x).any():
            print("NaN detected after fc2")
            return x

        x = self.dropout(x)
        x = self.fc3(x)
        if torch.isnan(x).any():
            print("NaN detected after fc3")
            return x

        x = self.sigmoid(x)
        if torch.isnan(x).any():
            print("NaN detected after sigmoid")
            return x

        return x    
    
if __name__ == "__main__":
    from dataset import BrainDataset
    from torch.utils.data import DataLoader, Dataset

    braindataset = BrainDataset(data_dir="/data2/om/ADNI dataset/data")
    dataloader = DataLoader(braindataset, batch_size=2, shuffle=True)

    # Example of how to initialize this class with Bottleneck blocks
    model = ResNetClassifier([3, 24, 36, 3], sample_input_D=32, sample_input_H=32, sample_input_W=32, num_seg_classes=2, shortcut_type='B').to('cuda')

    for data, labels in dataloader:
            output = model(data.to('cuda'))
            print(output)
            break