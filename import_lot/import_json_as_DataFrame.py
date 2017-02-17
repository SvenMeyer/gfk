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
array_size = (4,5)
# na = np.full(array_size, np.nan, np.float32)
na = np.zeros(array_size, np.float32)

idx_names = ['hhid','uid','cookieid']
# df = pd.DataFrame(columns=['hhid','uid','cookieid']) #, index=['hhid','uid','cookieid'])
# df = df.set_index(['hhid','uid','cookieid'])

# start import
print("start import..."),
start = time.time()

i = 0
col_names = {}
col_idx   = 0
row_names = {}
row_idx   = 0
row_names_df = pd.DataFrame(columns=idx_names)

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
            row_names_df[row_idx]={'hhid':d['hhid'] , 'uid':d['uid'] , 'cookieid':d['cookieid']}
            row_idx += 1      
        c = col_names[bh_id]
        r = row_names[cu_id]
        if d['featurevalue'] != 0:
            na[r,c] += 1
        
        # print(i, d['hhid'],d['uid'],d['cookieid'],d['featurekey'],d['featurevalue'])
        # make each line a real name-value dictionary
        # dict_nv = {'hhid':d['hhid'] , 'uid':d['uid'] , 'cookieid':d['cookieid'] , d['featurekey']:d['featurevalue']}
        # print(dict_nv)
        # df.loc()

        # print(df.loc['792611', '01', '12b31fa586482f2a9ca83b7c26b2ba8b'])

print("col_idx   = ", col_idx)
print("col_names = ", col_names)
print("row_idx   = ", row_idx)
for i in range(0,row_idx):
    print(row_names_df.loc[i] , na[i:])

# df = pd.DataFrame(na, index=row_names, columns=col_names) # DOES NOT WORK ... mixes rows !!!
print(df)

time_fit = (time.time() - start)
print(i, "lines processed")
print("DONE in ", time_fit, "sec")
