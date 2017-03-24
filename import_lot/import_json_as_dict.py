"""
Created on Sun Mar 12 21:00:00 2017

@author: sumeyer

import firehose data from name-value json

- uses defaultdict for data
- used dictionary for column names
- used dictionary for row/index names

STATUS : working
"""

import sys
import subprocess

def get_env():
    sp = sys.path[1].split("/")
    if "envs" in sp:
        return sp[sp.index("envs") + 1]
    else:
        return ""

def get_conda_env():
    # envs = subprocess.check_output('conda env list').splitlines()
    envs = subprocess.run(['conda','env','list'], stdout=subprocess.PIPE).stdout.splitlines()
    active_env = list(filter(lambda s: '*' in str(s), envs))[0]
    return str(active_env, 'utf-8').split()[0]

print("sys.path env = ", get_env())
# print("conda    env = ", get_conda_env())

import numpy as np
print("numpy.__version__ = ", np.__version__)

import pandas as pd
print("pandas.__version__ = ", pd.__version__)

import sklearn
print("sklearn.__version__ = ", sklearn.__version__)

import os
import time
import json

from collections import defaultdict
from sklearn.decomposition import PCA

TEST = False

COL_NAME_PID ="pnr"

# OPEN FILE

if TEST:
    HOME_DIR     = ""
    inp_dir  = ""
    filename = "test_sample"
else:
    HOME_DIR = "/media/sf_SHARE"
    if not os.path.isdir(HOME_DIR):
        HOME_DIR = os.path.expanduser("~")
    inp_dir = "/ML_DATA/GFK/S3/programmatic-dataprovider/data/de/training-datasets/v4/features.out.json/"
    filename = "part-r-00000-93628840-fd71-4a78-8bdb-6cafdf2b2738"
inp_ext  = "json"
datafile = HOME_DIR + inp_dir + filename + '.' + inp_ext
out_dir  = HOME_DIR + "/ML_DATA/GFK/DE/Lotame/"

print("open file : ", datafile)

# http://stackoverflow.com/questions/1602934/check-if-a-given-key-already-exists-in-a-dictionary
dd = defaultdict(lambda: 0)
# dd = dict()

start = time.time()
i = 0
with open(datafile) as f:
    lines = f.readlines()
    for line in lines:
        if line[0]=='{':         # if line does not start with curly bracket then it is comment or empty > ignore
            d = json.loads(line)
            i += 1
            key = (int(d['hhid']+d['uid']), d['cookieid'], d['featurekey'])
            if d['featurevalue'] != 0:
                # dd[d['hhid']+d['uid'], d['cookieid'], d['featurekey']] += 1
                if dd[key]<255:
                    # dd[key] = dd.get(key, 0) + 1
                    dd[key] += 1
            if i % 1e6 == 0:
                # break
                print(int(i/1e6), 'million lines processed in' , (time.time() - start), "sec")

time_fit = (time.time() - start)
print(i, 'lines processed in' , (time.time() - start), "sec")

print("create pandas.DataFrame from dict ... ", end='')
start = time.time()
df=pd.DataFrame(list(dd.keys()), columns=[COL_NAME_PID,'cookieid','featurekey'])
df['sum'] = pd.Series(list(dd.values()), dtype='uint8')
n_events = sum(df['sum'])
if i != n_events:
    print("WARNING - we have lost events !")
    print("sum(df['sum']) = ", n_events)
    
df_Lot = df.set_index([COL_NAME_PID, 'cookieid', 'featurekey'])['sum'].unstack(fill_value=0)
print("DONE in ", (time.time() - start), "sec")
n_events_table = df_Lot.sum(numeric_only=True).sum()
if n_events_table != n_events:
    print("ERROR - we have lost events !")
    print("df_Lot.sum(numeric_only=True).sum() = ", n_events_table)

# remove MultiIndex, set index to cookie column only
df_Lot = df_Lot.reset_index()

# count frequency per cookie
# df_Lot['freq'] = df_Lot.sum(axis=1)-df_Lot[COL_NAME_PID]

df_Lot_shape = df_Lot.shape
print("before checkig and removing duplicates - df.shape = ", df_Lot.shape)
if TEST:
    print(df_Lot)

# remove entries with same cookie (for different hhid-uid)
df_Lot.drop_duplicates(subset='cookieid', keep=False, inplace=True)
if df_Lot_shape != df_Lot.shape:
    print("removed rows with drop_duplicate cookies - df_Lot.shape = ", df_Lot.shape)

n_panel_unique = df_Lot[COL_NAME_PID].unique().shape[0]
print("n_panel_unique =", n_panel_unique)

if df_Lot.shape[0] != df_Lot['cookieid'].unique().shape[0]:
    print("WARNING: cookieid not unique")
    
df_Lot.drop('LEOCOOKIEFREQ', axis=1, inplace=True, errors='ignore') # errors='raise')
df_Lot.set_index(COL_NAME_PID, inplace=True)

print(df_Lot.iloc[:10, :5])

print("events in final table = ", df_Lot.sum(numeric_only=True).sum())
print("histogram : no of LOTBEH with 0 .. 99 panel visitor")
el = df_Lot.astype(bool).sum(axis=0)      # count no of non-zero entries for each LOTBEH
el[el<100].hist(bins=100, figsize=(10,10))
# drop LOTBET columns with very few entries (panel members)
MIN_panel_per_LOTBEH = 4
df_Lot.drop(el.index[el < MIN_panel_per_LOTBEH], axis=1, inplace=True)
print("removed LOTBEH columns with less than", MIN_panel_per_LOTBEH, "entries (panel members) - df_Lot.shape = ", df_Lot.shape)

ep = df_Lot.astype(bool).sum(axis=1)

output_file = out_dir + filename + '.pkl'
print("write pandas.DataFrame as picle file :", output_file, end=" ... ")
start = time.time()
df_Lot.to_pickle(output_file)
print("DONE in ", (time.time() - start), "sec")
# df = pd.read_pickle(file_name)

output_file = out_dir + filename + '.tsv'
print("write pandas.DataFrame as  tsv  file :", output_file, end=" ... ")
start = time.time()
df_Lot.to_csv(output_file, sep='\t')
print("DONE in ", (time.time() - start), "sec")

# store = pd.HDFStore(filename_out + '.h5')
# store['filename'] = df  # save it
# store['df']  # load it

first_feature_colname = df_Lot.columns[2]
na = df_Lot.ix[:, first_feature_colname:]
print("na.shape = ", na.shape)

'''
# max number of PCA components = nuber of features/colums
n_components_pca = min(na.shape[1],256)
print("start PCA with n_components =", n_components_pca)
start = time.time()
pca = PCA(n_components=n_components_pca)
pca.fit(na)
print("DONE in ", (time.time() - start), "sec")
pca_evr = pca.explained_variance_ratio_
print("pca.explained_variance_ratio_ = ")
print(pca_evr)
ps = pd.Series(pca_evr)
ps.plot()
ps.to_csv(out_dir + filename + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '_pca_evr.csv', sep='\t')
'''