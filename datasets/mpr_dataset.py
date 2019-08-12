from __future__ import print_function, division

from ast import literal_eval

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2


class MPR_Dataset(Dataset):
    LABELS_FILENAME = "new_labels.csv"

    ARTERY_COLUMN = "ARTERY_SECTION"
    VIEWPOINT_INDEX_COLUMN = "MPR_VIEWPOINT_INDEX"
    IMG_PATH_COLUMN = 'IMG_PATH'
    STENOSIS_SCORE_COLUMN = 'STENOSIS_SCORE'
    LABEL_COLUMN = 'LABEL'

    def __init__(self, root_dir, partition="train", transform=None, augmentation=None, config={}):
        self.root_dir = root_dir
        self.partition = partition
        self.config = config
        self.__load_data()
        self.__find_labels()
        self.transform = transform
        self.augmentation = augmentation

    def __load_data(self):
        df = pd.read_csv(os.path.join(self.root_dir, self.partition, self.LABELS_FILENAME))
        df = df[(df[self.ARTERY_COLUMN].isin(self.config['filters']["arteries"])) &
           (df[self.VIEWPOINT_INDEX_COLUMN] % self.config['filters']["viewpoint_index_step"] == 0)]
        df[self.STENOSIS_SCORE_COLUMN] = df[self.STENOSIS_SCORE_COLUMN].apply(literal_eval)
        self.df = df

    def __find_labels(self):
        mapper = {}
        for group, values in self.config['groups'].items():
            for value in values:
                mapper[value] = group
        self.labels = self.df[self.STENOSIS_SCORE_COLUMN].apply(lambda x: max([mapper[el] for el in x])).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]
        path = os.path.join(self.root_dir, self.partition, info[self.IMG_PATH_COLUMN])
        artery = info[self.ARTERY_COLUMN]
        stenosis_scores = info[self.STENOSIS_SCORE_COLUMN]
        y = self.labels[idx]
        X = cv2.imread(path)
        if transform:
            X = self.transform(X)
        return X, y

if __name__ == '__main__':
    # path_to_csv = 'labels.xlsx'
    root_dir = '../data/binary_classification_all_branches/'
    dataset_partition = 'train'
    transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    data_loader = MPR_Dataset(root_dir, dataset_partition, transform=transform)
    #
    # for img, labels in data_loader:
    #     print(labels, img.shape)
    #     break
