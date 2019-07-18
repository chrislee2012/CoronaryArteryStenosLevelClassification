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

from dataloaders.binary_data_loader import LAD_MPR_Loader
from dataloaders.sampler import ImbalancedDatasetSampler

from utils.training_functions import *

import warnings
warnings.filterwarnings("ignore")


# Load data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# train part
train_path_to_csv = '/home/petryshak/CoronaryArteryPlaqueIdentification/data/binary_classification_only_lad/train_without_25.csv'
train_path_to_data = '/home/petryshak/CoronaryArteryPlaqueIdentification/data/binary_classification_only_lad/'
dataset_partition = 'train'
lad_train = LAD_MPR_Loader(train_path_to_csv, train_path_to_data, dataset_partition)
train_loader = torch.utils.data.DataLoader(lad_train,
                                             batch_size=64, 
                                             # shuffle=True,
                                             sampler=ImbalancedDatasetSampler(lad_train),

                                            )
# val part
val_path_to_csv = '/home/petryshak/CoronaryArteryPlaqueIdentification/data/binary_classification_only_lad/val_without_25.csv'
val_path_to_data = '/home/petryshak/CoronaryArteryPlaqueIdentification/data/binary_classification_only_lad/'
dataset_partition = 'val'
lad_val = LAD_MPR_Loader(val_path_to_csv, val_path_to_data, dataset_partition)
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

# weights = [0.62566531, 2.48941134]
# class_weights = torch.FloatTensor(weights).cuda()
# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.Adam(model.parameters(), lr=0.001)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

data_loaders = {'train': train_loader, 'val': val_loader}
model = train_model(
                    model, data_loaders, criterion, 
                    optimizer_conv, exp_lr_scheduler,
                    'weights/pretrained_resnet18_balanced_data_without_25.pth', 
                    'pretrained_resnet50_balanced_data_without_25',
                    device,
                     num_epochs=20
                     )
