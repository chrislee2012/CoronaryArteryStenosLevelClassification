from __future__ import print_function, division

import time
from ast import literal_eval

import os
import random
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2


class MPR_Dataset(Dataset):
    LABELS_FILENAME = "labels.csv"

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
        df = df[
                    (df[self.ARTERY_COLUMN].isin(self.config['filters']["arteries"])) &
                    (df[self.VIEWPOINT_INDEX_COLUMN] % self.config['filters']["viewpoint_index_step"] == 0)
               ]
        df[self.STENOSIS_SCORE_COLUMN] = df[self.STENOSIS_SCORE_COLUMN].apply(literal_eval)
        self.df = df

    def __find_labels(self):
        mapper = {}
        for group, values in self.config['groups'].items():
            for value in values:
                mapper[value] = group
        self.labels = self.df[self.STENOSIS_SCORE_COLUMN].apply(lambda x: max([mapper[el] for el in x])).tolist()
        self.arteries = self.df[self.ARTERY_COLUMN].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]
        path = os.path.join(self.root_dir, self.partition, info[self.IMG_PATH_COLUMN])
        artery = self.arteries[idx]
        stenosis_scores = info[self.STENOSIS_SCORE_COLUMN]
        y = self.labels[idx]
        X = cv2.imread(path)

        if self.augmentation:
            X = self.augmentation(X)

        if self.transform:
            X = self.transform(X)
        return X, y


class MPR_DatasetSTENOSIS_REMOVAL(Dataset):
    LABELS_FILENAME = "labels.csv"

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
        df = df[
                    (df[self.ARTERY_COLUMN].isin(self.config['filters']["arteries"])) &
                    (df[self.VIEWPOINT_INDEX_COLUMN] % self.config['filters']["viewpoint_index_step"] == 0)
               ]
        df[self.STENOSIS_SCORE_COLUMN] = df[self.STENOSIS_SCORE_COLUMN].apply(literal_eval)
        self.df = df
        self.__filter_stenos()

    def __filter_stenos(self):
        mapper = \
            {
                'NORMAL': 0, '-': 0, 
                '25%': 1, '<25%': 1,
                '*50%': 2, '>50%': 2, '70%': 2, '50-70%': 2, '50%': 2, '>70%': 2, '90%': 2,  '90-100%': 2, 
                '75%': 2, '>75%': 2, '70-90%': 2, '>90%': 2, '25-50%': 2, '<50%': 2, '<35%': 2,
            }
        remove_elements = self.df[self.STENOSIS_SCORE_COLUMN].apply(lambda x: max([mapper[el] for el in x])).tolist()
        self.df = self.df[pd.Series(remove_elements, index=self.df.index) != 1]

    def __find_labels(self):
        mapper = {}
        for group, values in self.config['groups'].items():
            for value in values:
                mapper[value] = group
        self.labels = self.df[self.STENOSIS_SCORE_COLUMN].apply(lambda x: max([mapper[el] for el in x])).tolist()
        self.arteries = self.df[self.ARTERY_COLUMN].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]
        path = os.path.join(self.root_dir, self.partition, info[self.IMG_PATH_COLUMN])
        artery = self.arteries[idx]
        stenosis_scores = info[self.STENOSIS_SCORE_COLUMN]
        y = self.labels[idx]
        X = cv2.imread(path)

        if self.augmentation:
            X = self.augmentation(X)

        if self.transform:
            X = self.transform(X)
        return X, y

class MPR_Dataset_LSTM(Dataset):
    LABELS_FILENAME = "labels.csv"

    ARTERY_COLUMN = "ARTERY_SECTION"
    VIEWPOINT_INDEX_COLUMN = "MPR_VIEWPOINT_INDEX"
    IMG_PATH_COLUMN = 'IMG_PATH'
    STENOSIS_SCORE_COLUMN = 'STENOSIS_SCORE'
    LABEL_COLUMN = 'LABEL'
    SEGMENT_ID_COLUMN = "SEGMENT_ID"

    def __init__(self, root_dir, partition="train", level="img", transform=None, augmentation=None, config={}):
        self.root_dir = root_dir
        self.partition = partition
        self.config = config
        self.__load_data()
        self.__detect_segments()
        self.__find_labels()
        self.transform = transform
        self.augmentation = augmentation

    def __load_data(self):
        df = pd.read_csv(os.path.join(self.root_dir, self.partition, self.LABELS_FILENAME))
        if 'filters' in self.config:
            df = df[
                        (df[self.ARTERY_COLUMN].isin(self.config['filters']["arteries"])) &
                        (df[self.VIEWPOINT_INDEX_COLUMN] % self.config['filters']["viewpoint_index_step"] == 0)
                   ]
        df = df[~df['IMG_PATH'].str.contains('CTCALEK24101973/PLV_RCA/')]
        df[self.STENOSIS_SCORE_COLUMN] = df[self.STENOSIS_SCORE_COLUMN].apply(literal_eval)
        self.df = df

    def __detect_segments(self):
        self.df[self.SEGMENT_ID_COLUMN] = self.df[self.IMG_PATH_COLUMN].str.split("/").\
            apply(lambda x: x[-1].rsplit("_", maxsplit=1)[0]).factorize()[0]

    def __find_labels(self):
        mapper = {}
        for group, values in self.config['groups'].items():
            for value in values:
                mapper[value] = group
        self.df[self.LABEL_COLUMN] = self.df[self.STENOSIS_SCORE_COLUMN].apply(lambda x: max([mapper[el] for el in x]))
        self.labels = self.df.groupby(by=self.SEGMENT_ID_COLUMN)[self.LABEL_COLUMN].max().tolist()

    def __len__(self):
        # TODO: Add image length
        return len(self.df[self.SEGMENT_ID_COLUMN].unique())

    def __getitem__(self, idx):
        # TODO: Add image mask
        mask = self.df[self.SEGMENT_ID_COLUMN] == idx
        df_masked = self.df[mask]

        img_pathes = df_masked[self.IMG_PATH_COLUMN]
        images = []
        state = random.getstate()
        for img_path in img_pathes:
            img_path = os.path.join(self.root_dir, self.partition, img_path)
            img = cv2.imread(img_path)

            if self.augmentation:
                random.setstate(state)
                img = self.augmentation(img)

            if self.transform:
                img = self.transform(img)
            images.append(img)
        images = torch.stack(images)

        y = self.labels[idx]
        viewpoint_indexes = torch.tensor(df_masked[self.VIEWPOINT_INDEX_COLUMN][mask].tolist())
        X = images[viewpoint_indexes.argsort()]

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
    
    # for img, labels in data_loader:
    #     print(labels, img.shape)
    #     break
