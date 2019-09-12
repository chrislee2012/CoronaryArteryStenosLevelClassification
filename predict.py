from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
import os
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, \
                            precision_score, recall_score, auc, roc_auc_score, \
                            confusion_matrix, roc_curve
import yaml

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
sys.path.insert(0, 'datasets/')

from mpr_dataset import MPR_Dataset, MPR_Dataset_STENOSIS_REMOVAL, MPR_Dataset_LSTM, MPR_Dataset_H5
from ast import literal_eval


from os.path import join
import inspect
import importlib

import re
from os import listdir
import cv2

def calculate_metrics(col_section, col_ids, col_preds, col_labels, f1_average='macro'):
    """
    Calculate final auc and f1 metrics on three levels: per patient, per section and per artery
    :return: {dict} each metric as a key and its calculated metric as a value
    """
    assert len(col_section) == len(col_ids) == len(col_preds) == len(col_labels)

    metrics = {'ACC_section': 0, 'ACC_patient': 0, 'ACC_artery': 0, 'F1_section': 0, 'F1_patient': 0, 'F1_artery': 0}
    dict_artery = {'LAD': ['D-1', 'D-2', 'LAD', 'D-3', '2D-2', 'D-1Original', 'LADOriginal', 'D-4'],
                   'LCX': ['LCX', 'OM-2', 'OM-1', 'OM-3', 'OM', 'LCX-PLB', 'LCX-PDA', 'PLV_LCX', 'PDA_LCX'],
                   'RCA': ['RCA', 'RCA-PLB', 'RCA-PDA', 'PLV_RCA']}

    df = pd.concat([col_ids, col_section, col_preds, col_labels], axis=1)
    df = df.rename(columns={col_section.name: 'section', col_ids.name: 'patient', col_preds.name:
        'preds', col_labels.name: 'labels'})
    df['artery'] = df['section'].apply(lambda x: [k for k in dict_artery.keys() if x in dict_artery[k]][0])

    print(df.tail())
    # SECTION
    section_labels = df[['preds', 'labels', 'section', 'artery', 'patient']].groupby(['patient', 'section']).agg(lambda x: max(x))
    preds_section = df[['preds', 'labels', 'section', 'artery', 'patient']].groupby(['patient', 'section']).agg(lambda x: x.value_counts().index[0])
    acc = accuracy_score(preds_section['preds'], section_labels['labels'])
    f1 = f1_score(preds_section['preds'], section_labels['labels'], average=f1_average)
    metrics['ACC_section'], metrics['F1_section'] = acc, f1

    # ARTERY
    sect = section_labels.reset_index()
    artery_labels = sect.groupby(['patient', 'artery']).agg(lambda x: max(x))['labels']
    preds_artery = preds_section.reset_index().groupby(['patient', 'artery']).agg(lambda x: max(x))[
        'preds']  # x.value_counts().index[0])['preds']
    acc = accuracy_score(preds_artery, artery_labels)
    f1 = f1_score(preds_artery, artery_labels, average=f1_average)
    metrics['ACC_artery'], metrics['F1_artery'] = acc, f1

    # PATIENT
    art = artery_labels.reset_index()
    patient_labels = art.groupby(['patient']).agg(lambda x: max(x))['labels']
    #     print(preds_artery.reset_index())
    preds_patient = preds_artery.reset_index().groupby(['patient']).agg(lambda x: max(x))[
        'preds']  # x.value_counts().index[0])['preds']
    acc = accuracy_score(preds_patient, patient_labels)
    f1 = f1_score(preds_patient, patient_labels, average=f1_average)
    metrics['ACC_patient'], metrics['F1_patient'] = acc, f1

    return metrics

def __module_mapping(module_name):
    mapping = {}
    for name, obj in inspect.getmembers(importlib.import_module(module_name), inspect.isclass):
        mapping[name] = obj
    return mapping

def create_model(model_name, path_to_weigths, device):
    """
    Creates model and loads the weights.
    
    Possible values of the model_name:
    - LSTMClassification
    - LSTMDeepClassification
    - LSTMDeepResNetClassification
    """
    mapping = __module_mapping('models')
    model = mapping[model_name](n_classes=3)
    model.load_state_dict(torch.load(path_to_weigths))
    model.eval()
    model.to(device)
    return model

def create_dataloaders(config):
    root_dir = config["data"]["root_dir"]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    loaders = {partition: DataLoader(MPR_Dataset_H5(root_dir, partition=partition, config=config["data"], transform=transform), shuffle=False,
                batch_size=4) for partition in ["train", "val", "test"]}
    return loaders



def main():
    pass

if __name__ == '__main__':

    with open('config.yaml', 'r') as f:
       config = yaml.load(f, Loader=yaml.FullLoader)

    device = 'cuda'
    model = create_model('ShuffleNetv2', '/home/petryshak/CoronaryArteryPlaqueIdentification/experiments_major_vote/exp20/models/model_model_7_val_f1=0.5650427.pth', device)
    loaders = create_dataloaders(config)
    predictions = [] 

    for partition_name in ['test']:
        with torch.no_grad():
            
            for step, (x, y) in enumerate(tqdm(loaders[partition_name])):
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                _, predicted = torch.max(output, 1)

                # l = []
                # for i in range(len(predicted.cpu().detach().numpy())):
                #     for j in range(50):
                #         l.append(predicted[i].cpu().detach().item())
                # predictions.extend(l)
                # print(predicted.cpu().detach().numpy())
                predictions.extend(predicted.cpu().detach().numpy())

        p_test_df = pd.read_csv("/home/petryshak/CoronaryArteryPlaqueIdentification/data/all_branches_with_pda_plv_h5/{}/labels.csv".format(partition_name))

        print(len(predictions), p_test_df.shape)
        p_test_df = p_test_df[p_test_df['MPR_VIEWPOINT_INDEX']%1 == 0].reset_index()
        predictions = [int(x) for x in predictions]
        p_test_df['PRED'] = pd.Series(predictions)
        print(p_test_df['PRED'].isna().sum())
        # print(p_test_df['PRED'])
        p_test_df["STENOSIS_SCORE"] = p_test_df["STENOSIS_SCORE"].apply(literal_eval)
        p_test_df['PATIENT'] = p_test_df['IMG_PATH'].apply(lambda s: s.split('/')[1])
        mapper = {}
        for group, values in config['data']['groups'].items():
            for value in values:
                mapper[value] = group
        p_test_df["LABELS"] = p_test_df["STENOSIS_SCORE"].apply(lambda x: max([mapper[el] for el in x])).tolist()
        # p_test_df.to_excel('lol.xlsx')
        print(partition_name, ':')
        print(calculate_metrics(p_test_df['ARTERY_SECTION'], p_test_df['PATIENT'], p_test_df['PRED'], p_test_df['LABELS']))
        print()