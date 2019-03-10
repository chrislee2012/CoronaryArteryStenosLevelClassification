import os
import pandas as pd

def size(start_path):
    total_size = 0
    for path, dirs, files in os.walk(start_path):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)
    return total_size

def analysis(path):
    EXLUDE = ["$RECYCLE.BIN", "System Volume Information"]
    groups = os.listdir(path)
    total_size = 0
    info = dict(data_path=[], info_path=[], data_size=[])
    for group in groups:
        if group in EXLUDE:
            continue
        patients = os.listdir(os.path.join(path, group))
        for patient in patients:
            data_path = os.path.join(path, group, patient, "DICOMOBJ")
            total_size += size(data_path)
            files = os.listdir(os.path.join(path, group, patient))
            info_path = None
            for file in files:
                if file.endswith(".xlsx"):
                    info_path = os.path.join(path, group, patient, file)
            info["data_path"].append(data_path)
            info["info_path"].append(info_path)
            info["data_size"].append(size(data_path))

    # info = dict(size_GB=(total_size >> 30))
    return pd.DataFrame(info)

def info_parser(path):
    df = pd.read_excel(path, index_col=None, header=None)
    info = dict()
    rows = [list(row[row.notnull()]) for i, row in df.iterrows() if row.notnull().any()]
    print(rows)
    i = 0
    while i < len(rows):
        row = rows[i]
        if row[0] == "ID:":
            info["ID"] = row[1]
        elif row[0] == "DOB:":
            info["DOB"] = row[1]
        elif row[0] == "Age/Gender:":
            data = row[1].split()
            info["Age"] = int(data[0][:-2])
            info["Gender"] = data[1]
        i += 1
    return info

if __name__ == "__main__":
    # df = analysis("D:\\")
    # df.to_csv("data/info.csv")
    print(info_parser("D:\\CTCA WITH RECONS EXCEL REPORTS 321-350\\321 CTCADOP23111947\\REPORT DOP23111947.xlsx"))