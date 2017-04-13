import os
import glob
import pandas as pd
import numpy as np
from datetime import date
from datetime import timedelta

AddThis_DIR = "/media/sumeyer/WD_USB_4TB/SHARE/ML_DATA/GFK/AddThis/"
COL_NAME_FILE = "pig_header"
FILE_WILDCARD = "part-r-?????"


# read cookie file
cookie_filename = os.path.join(AddThis_DIR, "addthis_cookies.csv")
print("Reading cookie file :", cookie_filename)
df_cookies = pd.read_csv(cookie_filename, sep=',')

data_filename = "pixelview-turbo-no-porn-de.20170221-1100.0000"
data_filename = "pixelview-turbo-no-porn-de.20170221-????.????"

date_start = date(2017,2,21)
date_end   = date(2017,3,21)
dates = pd.date_range(date_start, date_end)
print("dates to be processed:", dates)


header_file = open(os.path.join(DATA_DIR, COL_NAME_FILE), 'r')
header_list = header_file.readline().split(';')

data_files = glob.glob(os.path.join(DATA_DIR,FILE_WILDCARD))


for f in filenames:
    target = open(f, 'w')
    
    
    target.write(line1)
    target.write("\n")


# version 1 : does not work if sep is specified
# df = pd.concat(map(pd.read_csv,          data_files))
# df = pd.concat(map(pd.read_csv(sep=';'), data_files))

# version 2 : short / non-verbose version
# df_from_each_file = (pd.read_csv(f, sep=';', header=None) for f in all_files)
# df   = pd.concat(df_from_each_file, ignore_index=True)

# version 3 : verbose

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
    print("number of dataframe columns = ", df.shape[1])
    print("number of column names      = ", len(header_list))

df.set_index(header_list[0], inplace=True)

df.to_csv(os.path.join(DATA_DIR, 'data.csv'), sep='\t')

print("DONE !")