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

# class LAD_MPR_Loader(Dataset):

#     def __init__(self, path_to_csv, root_dir, dataset_partition, augment=None):
#         self.labels = pd.read_csv(path_to_csv)
#         self.dataset_partition_name = dataset_partition # train test val
#         self.root_dir = root_dir
#         self.augment = augment
#         self.data_transformations = transforms.Compose([
#                 transforms.ToTensor(),
#                 # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ])
#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         # print(self.labels.iloc[idx]['PATIENT_NAME'])
#         artery_section_name = self._artery_secton_name_strip(self.labels.iloc[idx]['IMG_NAME'].split('_')[0])
#         patient_folder_name = self.labels.iloc[idx]['PATIENT_NAME']
#         image_name = self.labels.iloc[idx]['IMG_NAME']
#         y = torch.tensor(self.labels.iloc[idx]['LABEL'], dtype=torch.long)
#         stenosis_score = self.labels.iloc[idx]['STENOSIS_SCORE']
#         img_path = os.path.join(self.root_dir, self.dataset_partition_name, patient_folder_name, artery_section_name, image_name)
#         X =cv2.imread(img_path)# cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#         if self.augment:
#             X = self.augment(image=X)['image']
#         X = self.data_transformations(X)
#         return X, y, stenosis_score, patient_folder_name, self.labels.iloc[idx]['IMG_NAME']

#     def _artery_secton_name_strip(self, section_name):
#         if 'D-2' in section_name:
#             section_name = 'D-2'
#         elif 'D-1' in section_name:
#             section_name = 'D-1'
#         elif 'LAD' in section_name:
#             section_name = 'LAD'
#         return section_name

class Binary_MPR_Loader(Dataset):

    def __init__(self, root_dir, csv_name, dataset_partition, sample_each_nth_mpr_image=3, augment=None, transformations=None):
        self.labels = pd.read_excel(os.path.join(root_dir, dataset_partition, csv_name))
        self._sampe_nth_mpr_image(sample_each_nth_mpr_image)

        self.dataset_partition_name = dataset_partition # train test val
        self.root_dir = root_dir
        self.augment = augment
        self.data_transformations = transformations



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patient_folder_name = self.labels.iloc[idx]['PATIENT_NAME']
        image_name = self.labels.iloc[idx]['IMG_NAME']
        artery_section_name = self.labels.iloc[idx]['ARTERY_SECTION']
        stenosis_score = self.labels.iloc[idx]['STENOSIS_SCORE']

        y = torch.tensor(self.labels.iloc[idx]['LABEL'], dtype=torch.long)

        img_path = os.path.join(self.root_dir, self.dataset_partition_name, 'imgs', patient_folder_name, artery_section_name, image_name)
        X =cv2.imread(img_path)# cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if self.augment:
            X = self.augment(image=X)['image']

        X = self.data_transformations(X)
        return X, y #, stenosis_score, patient_folder_name, self.labels.iloc[idx]['IMG_NAME']

    def _sampe_nth_mpr_image(self, n_th):
        self.labels = self.labels[self.labels['MPR_VIEWPOINT_INDEX'] % n_th == 0]


if __name__ == '__main__':
    path_to_csv = 'train_labels_with_normal_and_minimal_stenosis_level.xlsx'
    root_dir = '../data/binary_classification_all_branches/'
    dataset_partition ='train'
    transforms = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    data_loader = Binary_MPR_Loader(root_dir, path_to_csv, dataset_partition, transformations=transforms)

    for img, labels in data_loader:
        print(labels, img.shape)
        break
