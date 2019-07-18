from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2

class LAD_MPR_Loader(Dataset):

    def __init__(self, path_to_csv, root_dir, dataset_partition, transform=None):
        self.labels = pd.read_csv(path_to_csv)
        self.dataset_partition_name = dataset_partition # train test val
        self.root_dir = root_dir
        self.data_transformations = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # print(self.labels.iloc[idx]['PATIENT_NAME'])
        artery_section_name = self._artery_secton_name_strip(self.labels.iloc[idx]['IMG_NAME'].split('_')[0])
        patient_folder_name = self.labels.iloc[idx]['PATIENT_NAME']
        image_name = self.labels.iloc[idx]['IMG_NAME']
        y = torch.tensor(self.labels.iloc[idx]['LABEL'], dtype=torch.long)
        stenosis_score = self.labels.iloc[idx]['STENOSIS_SCORE']
        img_path = os.path.join(self.root_dir, self.dataset_partition_name, patient_folder_name, artery_section_name, image_name)
        X =cv2.imread(img_path)# cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        X = self.data_transformations(X)
        return X, y, stenosis_score, patient_folder_name, self.labels.iloc[idx]['IMG_NAME']

    def _artery_secton_name_strip(self, section_name):
        if 'D-2' in section_name:
            section_name = 'D-2'
        elif 'D-1' in section_name:
            section_name = 'D-1'
        elif 'LAD' in section_name:
            section_name = 'LAD'
        return section_name


if __name__ == '__main__':
    path_to_csv = '/home/petryshak/CoronaryArteryPlaqueIdentification/data/binary_classification_only_lad/train_without_25.csv'
    path_to_data = '/home/petryshak/CoronaryArteryPlaqueIdentification/data/binary_classification_only_lad/'
    dataset_partition ='train'
    data_loader = LAD_MPR_Loader(path_to_csv, path_to_data, dataset_partition)
    for img, label, a in data_loader:
        print(label, a)
        break