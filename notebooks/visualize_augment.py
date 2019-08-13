from albumentations import (
    HorizontalFlip, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine, Flip, OneOf, Compose
)

from torch.utils.data import DataLoader
import pandas as pd
import os
from tqdm import tqdm
import re
from numpy import nan as Nan
import cv2
import yaml
import importlib
import inspect
from torchvision import transforms

from datasets.mpr_dataset import MPR_Dataset
from samplers import ImbalancedDatasetSampler, BalancedBatchSampler


def __module_mapping(module_name):
    mapping = {}
    for name, obj in inspect.getmembers(importlib.import_module(module_name), inspect.isclass):
        mapping[name] = obj
    return mapping


with open('../config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

mapping = __module_mapping('samplers')

sampler = mapping[config['dataloader']['sampler']]

transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

root_dir = config["data"]["root_dir"]
mapping_aug = __module_mapping('augmentations')
augmentation = mapping_aug[config['data']['augmentation']['name']](config['data']['augmentation']['parameters'])


train_dataset = MPR_Dataset(root_dir, partition="train", config=config["data"], transform=transform,
                            augmentation=augmentation)

train_loader = DataLoader(train_dataset, batch_size=8)

for step, (x, y) in enumerate(train_loader):
    print(x.shape)
    break