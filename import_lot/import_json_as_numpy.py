import os
import time
import json
import numpy as np
import pandas as pd

# OPEN FILE
home = os.path.expanduser("~")
# dir  = home + "/media/sumeyer/SSD_2/ML_DATA/programmatic-dataprovider/data/de/training-datasets/v4/features.out.json/"
# dir  = home + "/ML_DATA/programmatic-dataprovider/data/de/training-datasets/v4/features.out.json/"
dir  ="./"
# filename = "part-r-00000-93628840-fd71-4a78-8bdb-6cafdf2b2738.json"
filename = "test_sample.json"
datafile = dir + filename
print("open file : ", datafile)

# Create large array to (hopefully) fit all cookies (rows) x BehaviorIDs (columns)
array_size = (4096,32768)
array_size = (10,10)
# na = np.full(array_size, np.nan, np.float32)
na = np.zeros(array_size, np.uint8)

idx_names = ['hhid','uid','cookieid']
# df = pd.DataFrame(columns=['hhid','uid','cookieid']) #, index=['hhid','uid','cookieid'])
# df = df.set_index(['hhid','uid','cookieid'])

# start import
print("start import..."),
start = time.time()

i = 0
col_names = {} # type dictionary (unsorted !) = {'LTMBEH_1': 0, 'LTMBEH_3': 1, 'LTMBEH_2': 2}
col_idx   = 0
row_names = {} # type dictionary (unsorted !) = {('0011', '01', 'cookie_2'): 1, ('0022', '01', 'cookie_1'): 0}
row_idx   = 0

with open(datafile) as f:
    lines = f.readlines()
    for line in lines:
        d = json.loads(line)
        i += 1
        bh_id = d['featurekey']
        if bh_id not in col_names:
            col_names[bh_id] = col_idx
            col_idx += 1
        cu_id = (d['hhid'],d['uid'],d['cookieid'])
        if cu_id not in row_names:
            row_names[cu_id] = row_idx
            row_idx += 1      
        c = col_names[bh_id]
        r = row_names[cu_id]
        if d['featurevalue'] != 0:
            na[r,c] += 1
        

print("col_idx   = ", col_idx)
print("col_names = ", col_names)
print("row_idx   = ", row_idx)
for i,r in enumerate(row_names):
    print(i,r,na[i])


time_fit = (time.time() - start)
print(i, "lines processed")
print("DONE in ", time_fit, "sec")


print("create pandas.DatFrame from numpy-array, row_names and col_names ...")
start = time.time()
# create sorted lists of row_names and col_names by index
# http://pythoncentral.io/how-to-sort-python-dictionaries-by-key-or-value/
# > Custom sorting algorithms with Python dictionaries
df = pd.DataFrame(na[0:row_idx,0:col_idx], index=sorted(row_names, key=row_names.__getitem__), columns=sorted(col_names, key=col_names.__getitem__))
print("DONE in ", (time.time() - start), "sec")
print(df)
