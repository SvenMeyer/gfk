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

with open(datafile) as f:
    lines = f.readlines()
    for line in lines:
        d = json.loads(line)
        df.at[ (d['hhid'],d['uid'],d['cookieid']) , d['featurekey'] ] = 1

print(df)

time_fit = (time.time() - start)
print("DONE in ", time_fit, "sec")
