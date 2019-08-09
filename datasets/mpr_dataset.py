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


class MPR_Dataset(Dataset):
    LABELS_FILENAME = "labels.xlsx"

    ARTERY_COLUMN = "ARTERY_SECTION"
    VIEWPOINT_INDEX_COLUMN = "MPR_VIEWPOINT_INDEX"
    IMG_PATH_COLUMN = 'IMG_PATH'
    STENOSIS_SCORE_COLUMN = 'STENOSIS_SCORE'
    LABEL_COLUMN = 'LABEL'

    def __init__(self, root_dir, partition="train", transform=None, config={}):
        self.root_dir = root_dir
        self.partition = partition
        self.config = config
        self.df = pd.read_excel(os.path.join(root_dir, partition, self.LABELS_FILENAME))
        self.__filter()
        self.__find_labels()
        self.transform = transform

    def __find_labels(self):
        # TODO: make labels with apply pandas
        pass

    def __filter(self):
        df = self.df
        self.df = df[(df[self.ARTERY_COLUMN].isin(self.config["arteries"])) &
                     (df[self.VIEWPOINT_INDEX_COLUMN] % self.config["viewpoint_index_step"] == 0)]

    def __len__(self):
        return len(self.df)

    def __class(self, stenosis_scores):
        # TODO: find class
        pass

    def __getitem__(self, idx):
        info = self.df.iloc[idx]
        path = os.path.join(self.root_dir, self.partition, info[self.IMG_PATH_COLUMN])
        artery = info[self.ARTERY_COLUMN]
        stenosis_scores = info[self.STENOSIS_SCORE_COLUMN]
        y = self.__class(stenosis_scores)
        X = cv2.imread(path)
        if transform:
            X = self.transform(X)
        return X, y

if __name__ == '__main__':
    # path_to_csv = 'labels.xlsx'
    # root_dir = '../data/binary_classification_all_branches/'
    # dataset_partition ='train'
    transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    # data_loader = Binary_MPR_Loader(root_dir, path_to_csv, dataset_partition, transformations=transforms)
    #
    # for img, labels in data_loader:
    #     print(labels, img.shape)
    #     break
    config = {
        "arteries": ["LAD"],
        "viewpoint_index_step": 3
    }
    dataloader = Binary_MPR_Loader("../data/", tranform=transform, config=config)
