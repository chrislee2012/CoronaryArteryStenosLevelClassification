# CoronaryPlaqueIdentification
We utilize DNNs for identifying the level of stenosis in coronary arteries from CT scans and MPR images

### Dataset structure

The dataset has the following structure:

```
|-- RootDir
|     |-- Train
|           |-- train_labels_file.xlsx/.csv
|           |-- imgs
|                 |--  patient_1
|                       |-- LAD
|                       |-- RCA
|                       ...
|                 ...
|     |-- Val
|           |-- val_labels_file.xlsx/.csv
|           |-- imgs
|                 |--  val_patient_1
|                       |-- LAD
|                       |-- RCA
|                       ...
|                 ...
|     |-- Test
|           |-- test_labels_file.xlsx/.csv
|           |-- imgs
|                 |--  test_patient_1
|                       |-- LAD
|                       |-- RCA
|                       ...
|                 ...
|
```
