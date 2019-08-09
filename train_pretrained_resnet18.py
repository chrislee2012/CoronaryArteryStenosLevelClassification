import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataloaders.binary_data_loader import Binary_MPR_Loader
from dataloaders.sampler import ImbalancedDatasetSampler
from dataloaders.augmentation import light_aug, medium_aug, strong_aug

from utils.training_functions import *

import warnings
warnings.filterwarnings("ignore")


# Load data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define transform and aug
transforms = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
augmenatator = medium_aug()

ROOT_DIR = 'data/binary_classification_all_branches/'
SPECIFIC_ARTERY_SECTION = 'LAD'
train_csv_name = 'train_labels_with_normal_and_minimal_stenosis_level.xlsx'
val_csv_name = 'val_labels_with_normal_and_minimal_stenosis_level.xlsx'

# train part
dataset_partition = 'train'
lad_train = Binary_MPR_Loader(
                                ROOT_DIR, 
                                train_csv_name, 
                                dataset_partition,
                                specific_artery_section=SPECIFIC_ARTERY_SECTION,
                                augment=augmenatator, 
                                transformations=transforms
                              )

train_loader = torch.utils.data.DataLoader(lad_train,
                                             batch_size=64, 
                                             # shuffle=True,
                                             sampler=ImbalancedDatasetSampler(lad_train),

                                            )
# val part
dataset_partition = 'val'
lad_val = Binary_MPR_Loader(
                            ROOT_DIR, 
                            val_csv_name, 
                            dataset_partition,
                            specific_artery_section=SPECIFIC_ARTERY_SECTION,
                            transformations=transforms
                            )
val_loader = torch.utils.data.DataLoader(lad_val,
                                             batch_size=64,
                                             shuffle=False
                                            )

# Define the model
model = torchvision.models.resnet18(pretrained=True)
# model = torchvision.models.resnext50_32x4d(pretrained=True, progress=True)
# model = torchvision.models.resnet50(pretrained=True)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# PATH_TO_MODEL_WEIGHTS = 'weights/pretrained_resnext50_32x4d_balanced_data_without_25.pth'
# model.load_state_dict(torch.load(PATH_TO_MODEL_WEIGHTS))

model.to(device)

# weights = [1, 1.2]
# class_weights = torch.FloatTensor(weights).cuda()
# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.Adam(model.parameters(), lr=0.001)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

data_loaders = {'train': train_loader, 'val': val_loader}
model = train_model(
                    model, data_loaders, criterion, 
                    optimizer_conv, exp_lr_scheduler,
                    'weights_all_branches/minimum_stenosis_level_only_lad.pth', 
                    'minimum_stenosis_level_only_lad',
                    device,
                     num_epochs=20
                     )
