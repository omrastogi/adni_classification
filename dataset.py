import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split


class BrainDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.data = []
        self.labels = []
        self.mode = mode
        self.ad_dir = f"{data_dir}/AD_processed"
        for file in os.listdir(self.ad_dir):
            self.data.append(f"{self.ad_dir}/{file}")
            self.labels.append(1)
        
        self.cn_dir = f"{data_dir}/CN_processed"
        for file in os.listdir(self.cn_dir):
            self.data.append(f"{self.cn_dir}/{file}")
            self.labels.append(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        folder_path = self.data[idx]
        for file in os.listdir(folder_path):
            if 'mprage_brain.nii' in file:
                filepath = f"{folder_path}/{file}"
                break
        nii_file = nib.load(filepath)
        image_data = nii_file.get_fdata()
        image_tensor = torch.tensor(image_data, dtype=torch.float32)
        # resize_transform = Resize(spatial_size=(170, 256, 170))
        image_tensor_new = image_tensor.unsqueeze(0).unsqueeze(0)
        resized_tensor = F.interpolate(image_tensor_new, size=(170, 256, 170), mode='trilinear', align_corners=False)
        resized_tensor = resized_tensor.squeeze()
        return resized_tensor.unsqueeze(0), self.labels[idx]

    def create_train_test_split(self, test_size=0.1, random_state=42):
        data_train, data_test, labels_train, labels_test = train_test_split(self.data, self.labels, test_size=test_size, random_state=random_state)

        if self.mode == 'train':
            self.data = data_train
            self.labels = labels_train

        elif self.mode == "val":
            self.data = data_test
            self.labels = data_test 
