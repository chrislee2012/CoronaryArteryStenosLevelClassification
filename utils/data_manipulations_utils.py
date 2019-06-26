import os
import pandas as pd
from shutil import copyfile
import pydicom
from tqdm import tqdm
import re
from numpy import nan as Nan
import pydicom as dicom
import cv2

"""
========= COPY ONLY IMAGE DATA AND REPORTS WITHOUT DICOM VIEWER =========

EXAMPLE:
copy_dataset(patients_database, 'D:\\coronaryProject\\dataset')    

"""
def check_wether_patient_has_records(path_to_patient_folder, get_names_of_records=False):
    
    """
    Args:
        get_names_of_records (bool): wether return names of record files
        
    Returns:
        bool: Retuns value. True if patient folder contains the records and False otherwise. 
        or
        tuple (bool, list): returns bool and names of the record files for the patient.  
    """
    names_of_the_records = [x for x in os.listdir(path_to_patient_folder) if 'doc' in x or 'xlsx' in x]
    if get_names_of_records:
        return len(names_of_the_records) >=1, names_of_the_records
    else:
        return len(names_of_the_records) >=1

def check_wether_patient_has_image_data(path_to_patient_folder):
    """
    Returns:
        bool : Returns True if patient folder contatin image data and False otherwise
    """
    names_of_the_records = [x for x in os.listdir(path_to_patient_folder) if 'DICOMOBJ' in x]
    return len(names_of_the_records) >= 1

def get_structure_of_the_dataset(path_to_dataset):
    """
    
    Returns:
        dict :  keys - patient names(str): values (list of strings) - paths to the images and records
                                                    
    """
    patients_database = {}
    
    reports_folders = [x for x in os.listdir(path_to_dataset) if not any(i in x for i in
                            ['System', 'BIN', '$.BIN', 'Autorun.inf','Seagate', 'SeagateExpansion.ico', 
                             'Start_Here_Mac.app', 'Start_Here_Win.exe', 'Warranty.pdf'])]
                       #'System' not in x and 'BIN' not in x]
    for report_folder in tqdm(reports_folders):
        patients_per_folder = os.listdir(os.path.join(path_to_dataset, report_folder))
        
        for patient in patients_per_folder:
            
            files_in_patient_folder = os.listdir(os.path.join(path_to_dataset, report_folder, patient))
            
            if check_wether_patient_has_image_data(os.path.join(path_to_dataset, report_folder, patient)):
                patient_images = os.listdir(os.path.join(path_to_dataset, report_folder, patient, 'DICOMOBJ'))
                patient_images_paths = [os.path.join(path_to_dataset, report_folder, patient, 'DICOMOBJ', x) 
                                 for x in patient_images]
            else:
                patient_images = []
                patient_images_paths = []
            _, patient_records = check_wether_patient_has_records(
                                      os.path.join(path_to_dataset, report_folder, patient), 
                                      get_names_of_records=True)
            patient_records_paths = [os.path.join(path_to_dataset, report_folder, patient, x) for x in patient_records]
            patients_database[patient] = []
            patients_database[patient] += patient_records_paths
            patients_database[patient] += patient_images_paths
    
    return patients_database

def copy_dataset(patients_database, path_to_copy):
    """
    Copy only image data and records without DICOM viewer program
    Args:
        patients_database (dict): dictionary with patients and corresponding 
                                  images and records
        path_to_copy (str): destination folder, where all dataset will
                            be located
        
    Returns:
        None
    """
    # Create folder to the dataset
    if not os.path.exists(path_to_copy):
        os.mkdir(path_to_copy)
    
    for patient in tqdm(patients_database):
        # Check wether patient's folder contains images
        if len(patients_database[patient]) <=2:
            continue
            
        # Check wether patient contains the records
        path_to_the_patient = patients_database[patient][0]
        path_to_the_patient = '\\'.join(path_to_the_patient.split('\\')[:2])
        if not check_wether_patient_has_records(path_to_the_patient):
            continue
         
        group_folder_name = patients_database[patient][0].split('\\')[0][2:]
        group_folder_name = '_'.join([x.lower() for x in group_folder_name.split()])
        patient_folder_name = patients_database[patient][0].split('\\')[1]
        patient_folder_name = '_'.join([x for x in patient_folder_name.split()])
        
        # Create directories
        if not os.path.exists(os.path.join(path_to_copy, group_folder_name)):
            os.mkdir(os.path.join(path_to_copy, group_folder_name))
        if not os.path.exists(os.path.join(path_to_copy, group_folder_name, patient_folder_name)):
            os.mkdir(os.path.join(path_to_copy, group_folder_name, patient_folder_name))
        
        # Copy Records
        shutil.copy(patients_database[patient][0], os.path.join(
                path_to_copy, group_folder_name, patient_folder_name, patients_database[patient][0].split('\\')[-1]))
        
        # Create folder patients's for images
        if not os.path.exists(os.path.join( path_to_copy, group_folder_name, patient_folder_name, 'images')):
            os.mkdir(os.path.join( path_to_copy, group_folder_name, patient_folder_name, 'images'))
            
        # Copy images
        for i in range(1, len(patients_database[patient])):
            shutil.copy(patients_database[patient][i], os.path.join(
                path_to_copy, group_folder_name, patient_folder_name, 'images', patients_database[patient][i].split('\\')[-1]))

"""
========= COPY ONLY MPR IMAGES FROM THE DATASET =========

EXAMPLE:
copy_mpr_images_per_patient(r'E:\\', 'path_to_save')
"""

def copy_mpr_records(path_to_dataset, path_to_save):
    """
    Copy all records from the dataset to path_to_save folder.
    """
    folders = [x for x in os.listdir(path_to_dataset) if 'WITH RECONS ' in x]

    for folder_name in folders:
        for patient_name in os.listdir(os.path.join(path_to_dataset, folder_name)):
            files = os.listdir(os.path.join(path_to_dataset, folder_name, patient_name))
            files = [x for x in files if ('xlsx' in x) or ('doc' in x)]
            files = files[0] if len(files)>0 else None
            if files:
                copyfile(os.path.join(path_to_dataset, folder_name, patient_name, files), os.path.join(path_to_save, files))

def copy_mpr_images_per_patient(path_to_dataset, path_to_save):
    """
    Creates in path_to_save folder for each patient, where all MPR DICOM files are located.
    """
    folders = [x for x in os.listdir(path_to_dataset) if 'WITH RECONS ' in x]

    unique_modalities = []
    raw_images = ['CTCA', 'CALCIUM SCORE', 'Scout', 'AW electronic film', '40', '81', '85']
    for folder_name in tqdm(folders):
        for patient_name in tqdm(os.listdir(os.path.join(path_to_dataset, folder_name))):
            patient_files = os.listdir(os.path.join(path_to_dataset, folder_name, patient_name))

            # check wether patient contains the DICOMOBJ folder, skip the patient if the images were missed
            patient_files = [x for x in patient_files if 'DICOMOBJ' in x]
            if patient_files:
                images = list(os.walk(os.path.join(path_to_dataset, folder_name, patient_name, 'DICOMOBJ')))[0][2]
            else:
                continue

            for image_name in images:
                dicom_obj = pydicom.dcmread(os.path.join(path_to_dataset, folder_name,patient_name, 'DICOMOBJ', image_name))
                if dicom_obj.SeriesDescription not in unique_modalities:
                    unique_modalities.append(dicom_obj.SeriesDescription)
                    print(unique_modalities)
                pass
    print(unique_modalities)


"""
DICOM MANIPULATIONS

"""

def split_mpr_name(mpr_name):
    '''
    Removes the noise in the MPR names

    Args:
        - mpr_name(str): name of the mpr type in the DICOM file

    Returns:
        - str: mpr name without noise
    '''
    return \
        "".join(mpr_name.split()).replace('*', '').replace('original', '') \
        .replace('LIMA-', '').replace('Branchof','').replace('TOPDA', '').replace('PDATO', '')

def get_patient_dictionary(path_to_patient_folder):
    """
    
    Returns dict of different types of images in the folder of patient. 
    
    Returns:
        dict: key - type of images(specific MPR type, raw CT, 3d model etc...); value - list of DICOM files, which sorted in the ascending order with restepct to the depth of the image slice.
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

def show_stack(list_of_imgs, rows=6, cols=6, start_with=10, show_every=3):
    """
    Show stack of the images with the given parameters.
    """
    fig,ax = plt.subplots(rows,cols,figsize=[20,20])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(list_of_imgs[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()


"""
RECORDS MANIPULATIONS

"""
def strip_mpr_lad_name(mpr_name):
    """
    Strip MPR name of the LAD artery. We do this step because the name in the doctor's report 
    is not equal to the name in the MPR. 
    
    Returns:
        - str: striped string
    """
    return "".join(mpr_name.split()).replace('*', '').replace('original', '')

def read_and_strip_record(path_to_record):
    '''
    Read record file and remove empty rows and rows with all NaNs.
    
    Returns:
        - Pandas DataFrame: 
    '''
    excel_file = pd.read_excel(path_to_record,index_col=None, header=None)
    excel_file.dropna(how='all')
    excel_file.rename(columns={0: 'a', 1: 'b'}, inplace=True)
    excel_file = excel_file.fillna('  ')
    excel_file = excel_file.replace('', '  ', regex=True)
    excel_file = excel_file.drop(excel_file[excel_file['a'].str.isspace()].index)
    return excel_file

def get_lad_info_from_report(striped_record, artery_type):
    """
    Takes striped(without any empty lines and NaNs) and returns info only about the certain artery type. 
    
    Returns:
        - list: each element is the string with some info about certain artery type
    """
    lad_info = []
    wether_add = False
    lad_info.append(striped_record.iloc[0]['b'])
    for ind, row_value in striped_record.iterrows():
        
        if wether_add and row_value['a'].isupper():
            break
        if wether_add:
            lad_info.append(row_value['a'])
        
        if artery_type in row_value['a']:
            wether_add = True
    return lad_info

def get_level_of_stenosis_from_string(artery_info):
    """
    Args:
        - artery_info(list of strings): each element is the string with some info about certain artery type
    Returns:
        - list of str: each element is the string with percentage of stenosis. 
    """
    return [x.strip() for x in re.findall(r'.\d{1,3}.?\d{1,3}\%', artery_info)]