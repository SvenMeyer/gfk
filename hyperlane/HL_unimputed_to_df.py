import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = "/media/sf_SHARE/ML_DATA/GFK/DE/Hyperlane/unimputed-target-groups/2017-02-01/"
COL_NAME_FILE = "pig_header"
FILE_WILDCARD = "part-r-?????"

# does not work if sep is specified
# df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(DATA_DIR,FILE_WILDCARD))))
# df = pd.concat(map(pd.read_csv(sep=';'), glob.glob(os.path.join(DATA_DIR,FILE_WILDCARD))))

header_file = open(os.path.join(DATA_DIR, COL_NAME_FILE), 'r')
header_list = header_file.readline().split(';')

data_files = glob.glob(os.path.join(DATA_DIR,FILE_WILDCARD))

# short / non-verbose version
# df_from_each_file = (pd.read_csv(f, sep=';', header=None) for f in all_files)
# df   = pd.concat(df_from_each_file, ignore_index=True)

df = pd.DataFrame()
for filename in data_files:
    print("start processing file : ", filename, end='')
    df_new = pd.read_csv(filename, sep=';', header=None, dtype=np.int32)
    print(df_new.shape, end='')
    df = df.append(df_new, ignore_index=True)
    print(df.shape)

if len(header_list) == df.shape[1]:
    df.columns = header_list
else:
    print("ERROR")
    print("dataframe.shape = ", df.shape)
    print("number of column names = ", len(header_list))

df.set_index(header_list[0], inplace=True)

df.to_csv(os.path.join(DATA_DIR, 'data.csv'), sep='\t')

print("DONE !")