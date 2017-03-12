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

# OPEN FILE
home    = os.path.expanduser("~")
inp_dir = "/ML_DATA/gfk/S3/programmatic-dataprovider/data/de/training-datasets/v4/features.out.json/"
filename = "part-r-00000-93628840-fd71-4a78-8bdb-6cafdf2b2738"
inp_ext  = "json"
datafile = home + inp_dir + filename + '.' + inp_ext
# datafile = "./test_sample2.json"

out_dir = "/ML_DATA/gfk/DE/"

print("open file : ", datafile)

# http://stackoverflow.com/questions/1602934/check-if-a-given-key-already-exists-in-a-dictionary
# dd = defaultdict(lambda: 0)
dd = dict()

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
                dd[key] = dd.get(key, 0) + 1
            if i % 1e6 == 0:
                # break
                print(int(i/1e6), 'million lines processed in' , (time.time() - start), "sec")

time_fit = (time.time() - start)
print(i, 'lines processed in' , (time.time() - start), "sec")

print("create pandas.DataFrame from dict ... ", end='')
start = time.time()
df=pd.DataFrame(list(dd.keys()), columns=['hhid-uid','cookieid','featurekey'])
df['sum'] = pd.Series(list(dd.values()))
df_table = df.set_index(['hhid-uid','cookieid','featurekey'])['sum'].unstack(fill_value=0)
print("DONE in ", (time.time() - start), "sec")

# remove MultiIndex, set index to cokkie colum only
df_table = df_table.reset_index()

df_table_shape = df_table.shape
print("before checkig and removing duplicates - df.shape = ", df_table.shape)
# remove entries with same cookie (for different hhid-uid)
df_table.drop_duplicates(subset='cookieid', keep=False, inplace=True)
if df_table_shape != df_table.shape:
    print("removed rows with drop_duplicate cookies - df_table.shape = ", df_table.shape)

n_panel_unique = df_table['hhid-uid'].unique().shape[0]
print("n_panel_unique =", n_panel_unique)

if df_table.shape[0] != df_table['cookieid'].unique().shape[0]:
    print("WARNING: cookieid not unique")
    
# print first 10 lines, first 6 columns
print(df_table.iloc[:10,[0,1,2,3,4,5]])

print("write pandas.DataFrame as picle file ...")
start = time.time()
df_table.to_pickle(home + out_dir + filename + '_' + time.strftime("_%Y-%m-%d_%H-%M-%S") + '.pkl')
print("DONE in ", (time.time() - start), "sec")
# df = pd.read_pickle(file_name)
'''
print("write pandas.DataFrame as csv file ...")
start = time.time()
df_data.to_csv(home + out_dir + filename + '_' + time.strftime("_%Y-%m-%d_%H-%M-%S") + '.csv', sep='\t')
print("DONE in ", (time.time() - start), "sec")
'''
# store = pd.HDFStore(filename_out + '.h5')
# store['filename'] = df  # save it
# store['df']  # load it

first_feature_colname = df_table.columns[3]
na = df_table.ix[:,first_feature_colname:]
print("na.shape = ", na.shape)

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

ps.to_csv(home + out_dir + filename + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '_pca_evr.csv', sep='\t')