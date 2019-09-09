import pandas as pd
import pydicom
import pydicom as dicom
from PIL import Image
import numpy as np
import os
import cv2
import shutil
# from __future__ import print_function
from tqdm import tqdm
from shutil import copyfile
from numpy import nan as Nan
from ast import literal_eval
import h5py
import cv2
import re


def map_mpr_name_to_record_name(mpr_name):
    main_branches_dict = {
        'LAD': ['LAD', 'LAD ', 'LAD Original', 'LAD original', 'LAD *', 'LAD*'],
        'D-1':['LAD-D1 original', 'LAD-D1 Original', 'LAD-D1', 'LAD-D1 *', 'LAD -D1', 'LAD -D1', 'LAD - D1', 'D1'],
        'D-2':['LAD-D2', 'LAD-D2 *', 'LAD-D2', '2LAD-D2', 'LAD -D2', 'LAD-D2 original', 'LAD -D2'],
        'D-3': ['LAD-D3', 'LAD-D3 *', 'LAD-D3', 'LAD-D3 original', ],
        'D-4': [ 'LAD - D4 *', 'LAD-D4', 'LAD-D4 *'],
        'RCA': ['RCA', 'RCA *', 'RCA*', 'RCA original'],
        'OM':['OM*', 'LCX-OM  *', 'OM *', 'OM', 'LCX-OM*', 'LCX - OM *', 'LCX-OM original', 'LCX-OM *', 'LCX-OM', 'OM original'],
        'OM-1': ['LCX-OM1 *', 'OM1 *', 'OM1', 'LCX-OM1', 'LCX -OM1 *', 'LCX-OM1*'],
        'OM-2': ['LCX-OM2 *', 'OM2 *', 'LCX-OM2', 'LCX - OM2 *', 'LCX -OM2 *', 'OM2*', 'LCX-OM2*'],
        'OM-3': ['LCX-OM3 *', 'LCX -OM3 *', 'OM3',  'LCX-OM3*', 'LCX-OM3', 'OM3 *', 'OM3*'],
        'OM-4': ['OM4 *', 'OM4', 'LCX-OM4 *'],
        'LCX': ['LCX', 'LCX *', 'LCX original', 'LCX  *', 'LCX*'],
        'PDA_RCA': ['RCA-PDA','RCA -PDA', 'RCA-PDA*', 'RCA-PDA *', 'RCA-PDA1','RCA-PDA2', 'RCA-PDA2 *','RCA-PDA2', 
                    'RCA-PDA2*'],
        'PLV_RCA': ['RCA-PLB', 'RCA-PLB ', 'RCA-PLB', 'RCA -PLB*', 'RCA-PLB1 *','RCA-PLB1', 'RCA-PLB1 *','RCA-PLB2', 
                    'RCA-PLB2 *'],
        'PDA_LCX': ['LCX-PDA *', 'LCX-PDA', 'LCX-PDA2', 'LCX-PDA2 *'],
        'PLV_LCX': ['LCX-PLB', 'LCX-PLB *', 'LCX-PLB1', 'LCX-PLB2',  'LCX-PLB2 *'],
        'THRASH': ['PLB  *', 'PLB *', 'PLB original', 'PLB','PLB*','PLB1 *', 'PLB1*', 'PLB1','PLB2 *', 'PLB2*', 'PLB2']
    }
    
    for key in main_branches_dict:
        if mpr_name in main_branches_dict[key]:
            return key

def split_mpr_name(mpr_name):
    return \
        "".join(mpr_name.split()).replace('*', '').replace('original', '') \
        .replace('LIMA-', '').replace('Branchof','').replace('TOPDA', '').replace('PDATO', '')

def get_patient_dictionary(path_to_patient_folder):
    """
    
    Returns dict of different types of images in the folder of patient. 
    
    Returns:
        dict: key - type of images; value - list of DICOM files, which sorted in the ascending order with restepct to the
                    depth of the image slice.
    """
    patient_dict = {}
    
    dicom_file_names = os.listdir(path_to_patient_folder)
    
    for i in range(len(dicom_file_names)):
        cur_dicom_obj = dicom.dcmread(os.path.join(path_to_patient_folder, dicom_file_names[i]))
        
        if cur_dicom_obj.SeriesDescription not in patient_dict.keys():
            patient_dict[cur_dicom_obj.SeriesDescription] = []
        patient_dict[cur_dicom_obj.SeriesDescription].append(cur_dicom_obj)
        
    # sort each type of images with respect to their depth in ascending order
    for i in patient_dict:
        patient_dict[i].sort(key=lambda x: x.InstanceNumber)
    
    return patient_dict

def get_pixels_hu(list_of_imgs):
    """
    Convert stack of the images into Houndsfeld units
    """
    image = np.stack([s.pixel_array for s in list_of_imgs])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = list_of_imgs[0].RescaleIntercept
    slope = list_of_imgs[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def remove_text(img):
    mask = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)[1][:,:,0]
    dilated_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    dst = cv2.inpaint(img, dilated_mask, 5, cv2.INPAINT_NS)
    return dst

def save_img(img, img_name):
    with h5py.File(img_name, 'w') as hf: 
        Xset = hf.create_dataset \
                (
                    name='X',
                    data=img,
                    shape=(img.shape[0], img.shape[1]),
                    maxshape=(img.shape[0], img.shape[1]),
                    compression="lzf",
                )

PATH_TO_DATA = '/home/petryshak/CoronaryArteryPlaqueIdentification/data/ONLY_MPR'

df2 = pd.read_excel('ExtractLabels/lad_rca_lcx.xlsx')

df2.drop(['Unnamed: 0', 'LAD', 'D-1', 'D-2', 'D-3','RCA', 'LCX', 'OM', 'OM-1', 'OM-2', 'OM-3'],
         axis=1, inplace=True)


# path_to_data = r'D:\coronaryProject\dataset\binary_classification_MPR\images'
# path_to_new_data = r'E:\ONLY_LAD\\'
path_to_data = r'/home/petryshak/CoronaryArteryPlaqueIdentification/data/ONLY_MPR'
path_to_new_data = r'/home/petryshak/CoronaryArteryPlaqueIdentification/data/all_branches_with_pda_plv_h5'
list_of_patients = os.listdir(path_to_data)
# patient_dictionary = get_patient_dictionary(path_to_data + '\\'+ list_of_patients[0])

for i in tqdm(range(len(list_of_patients))):
    patient_dictionary = get_patient_dictionary(path_to_data + '/'+ list_of_patients[i])
    splited_mpr_names_filtered = [map_mpr_name_to_record_name(x) for x in patient_dictionary.keys()]
    dict_keys = list(patient_dictionary.keys())
    
    # change keys in the dict to the corresponding labels in the reports
    for k in range(len(dict_keys)):
        if splited_mpr_names_filtered[k]:
            if dict_keys[k] in splited_mpr_names_filtered:
                pass
            else:
                patient_dictionary[splited_mpr_names_filtered[k]] = patient_dictionary[dict_keys[k]]
                del patient_dictionary[dict_keys[k]]
        else:
            del patient_dictionary[dict_keys[k]]
#     if i > 5:
#         break
    if not os.path.exists(os.path.join(path_to_new_data, list_of_patients[i])):
        os.mkdir(os.path.join(path_to_new_data, list_of_patients[i]))
        

    if list_of_patients[i] in [x for x in list_of_patients if 
             x.split(' ')[1] not in list(df2['REPORT_ID']) and
             x.split(' ')[1] not in list(df2['PATIENT_ID']) and 
             x not in list(df2['REPORT_ID']) and
             x not in list(df2['PATIENT_ID'])
            ]:
        continue

    if list_of_patients[i].split(' ')[1] in list(df2['REPORT_ID']):
        patient_row = df2[df2['REPORT_ID'] == list_of_patients[i].split(' ')[1]]
    elif  list_of_patients[i].split(' ')[1] in list(df2['PATIENT_ID']):
        patient_row = df2[df2['REPORT_ID'] == list_of_patients[i].split(' ')[1]]
    elif list_of_patients[i] in df2['REPORT_ID']:
        patient_row = df2[df2['REPORT_ID'] == list_of_patients[i]]
    else:# list_of_patients[i] in df2['PATIENT_ID']:
        patient_row = df2[df2['PATIENT_ID'] == list_of_patients[i]]
    
    if not patient_row.empty:
        if 'THRASH' in patient_dictionary.keys():
            if patient_row['PLV_RCA'].iloc[0] != '-':
                patient_dictionary['PLV_RCA'] = patient_dictionary['THRASH']
            elif patient_row['PLV_LCX'].iloc[0] != '-':
                patient_dictionary['PLV_LCX'] = patient_dictionary['THRASH']
    if 'THRASH' in patient_dictionary.keys():
        del patient_dictionary['THRASH']


    for key in patient_dictionary.keys():
        for dicom_file in patient_dictionary[key]:
            if not os.path.exists(os.path.join(path_to_new_data, list_of_patients[i])):
                os.mkdir(os.path.join(path_to_new_data, list_of_patients[i]))
            
            if not os.path.exists(os.path.join(path_to_new_data, list_of_patients[i], key)):
                os.mkdir(os.path.join(path_to_new_data, list_of_patients[i], key))
                
            cur_img = dicom_file.pixel_array
            cur_img[cur_img == -2000] = 0
            intercept = dicom_file.RescaleIntercept
            slope = dicom_file.RescaleSlope
            if slope != 1:
                cur_img = slope * cur_img.astype(np.float64)
                cur_img = cur_img.astype(np.int16)

            cur_img += np.int16(intercept)

            final_result = np.array(cur_img, dtype=np.int16)
            save_img(final_result, 
                     os.path.join(
                                    path_to_new_data, 
                                    list_of_patients[i], 
                                    key,
                                    list_of_patients[i]+'_'+str(dicom_file.InstanceNumber)+'.h5'
                                  )
                     )

#             cv2.imwrite(os.path.join(path_to_new_data, 
#                                             list_of_patients[i], 
#                                             key,
#                                             list_of_patients[i]+'_'+str(dicom_file.InstanceNumber)+'.png'
#                                            ),
#                         cv2.normalize(dicom_file.pixel_array, None, alpha = 0, 
#                                       beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
#                        )