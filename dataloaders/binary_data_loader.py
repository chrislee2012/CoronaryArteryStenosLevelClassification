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

    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                        self.labels.iloc[idx, 0])

        image = io.imread(img_name)
        label = self.labels.iloc[idx, 1]
        return image, label

if '__name__' == '__main__':
    loader = LAD_MPR_Loader('lad_records.csv', 'images/')
    for i in loader:
        pass