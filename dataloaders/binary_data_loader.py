from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class LAD_MPR_Loader(Dataset):

    def __init__(self, path_to_csv, root_dir, transform=None):
        self.labels = pd.read_csv(path_to_csv)
        self.dataset_partition_name = path_to_csv.split('/')[-1].split('.')[0]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        artery_section_name = self.labels.iloc[idx, 2].split('_')[0]
        patient_folder_name = self.labels.iloc[idx, 1]
        image_name = self.labels.iloc[idx, 2]
        y = self.labels.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, self.dataset_partition_name, patient_folder_name, artery_section_name, image_name)
        image = io.imread(img_path)
        return image, y

if __name__ == '__main__':
    path_to_csv = '/home/petryshak/CoronaryArteryPlaqueIdentification/data/binary_classification_only_lad/train.csv'
    path_to_data = '/home/petryshak/CoronaryArteryPlaqueIdentification/data/binary_classification_only_lad/'
    data_loader = LAD_MPR_Loader(path_to_csv, path_to_data)
    for img, label in data_loader:
        print(img.shape, label)
        break