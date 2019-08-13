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
import scipy
from datasets.mpr_dataset import MPR_Dataset
from samplers import ImbalancedDatasetSampler, BalancedBatchSampler
from PIL import Image
import matplotlib.pyplot as plt


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
augmentation = mapping_aug[config['data']['augmentation']['name']](**config['data']['augmentation']['parameters'])

train_dataset = MPR_Dataset(root_dir, partition="train", config=config["data"], transform=transform) #,
                            # augmentation=augmentation)

train_loader = DataLoader(train_dataset, batch_size=8)

path_temp_aug = './tempaugm'
im_id = 0
for step, (x, y) in enumerate(train_loader):
    if not os.path.exists(path_temp_aug):
        os.mkdir(path_temp_aug)
    while im_id < 100:
        imgs = x.numpy()
        for i in imgs:
            # im = Image.fromarray(i.reshape(512, 512, 3))
            # im.save(os.path.join(path_temp_aug,'{}.png'.format(im_id)))
            plt.imshow(i[0, :, :], cmap='gray')
            plt.savefig(os.path.join(path_temp_aug,'{}.png'.format(im_id)))
            im_id += 1
    else:
        break