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
filename = "part-r-00000-93628840-fd71-4a78-8bdb-6cafdf2b2738.json"
# filename = "test_sample.json"
datafile = dir + filename
print("open file : ", datafile)

idx_names = ['hhid','uid','cookieid']
df = pd.DataFrame(columns=idx_names)
df.set_index(idx_names,inplace=True)

# start import
print("start import..."),
start = time.time()

i = 0
with open(datafile) as f:
    lines = f.readlines()
    for line in lines:
        d = json.loads(line)
        # if d['featurekey'] in df.columns:
        df.at[ (d['hhid'],d['uid'],d['cookieid']) , d['featurekey'] ] = 1
        i += 1
        if i % 1000 == 0: # only read first 1000 lines of file
            break

time_fit = (time.time() - start)
print("DONE in ", time_fit, "sec")

print("df.iloc[:5,:5] = ")
print(df.iloc[:5,:5])

# replace NaN by 0
df.fillna(value=0, axis='columns', inplace=True)

print("df.index.has_duplicates = ", df.index.has_duplicates)

print("generating statistics ...")
print(df.describe(include='all'))

filename_out = "DEtdv4_df_bin_slice_1000"

df.to_pickle(dir + filename_out + '.pkl')  # where to save it, usually as a .pkl
# df = pd.read_pickle(file_name)

df.to_csv(dir + filename_out + '.csv', sep='\t')

store = pd.HDFStore(filename_out + '.h5')
store['bin_v1'] = df  # save it
# store['df']  # load it