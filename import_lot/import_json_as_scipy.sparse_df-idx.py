'''
import firehose data from name-value json

- uses numpy array for data
- used dictionary for column names
- used dictionary for row/index names

STATUS : working

PROBLEM: ugly code, performance is not so good

'''
import os
import time
import json
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

# OPEN FILE
home    = os.path.expanduser("~")
inp_dir = "/ML_DATA/gfk/S3/programmatic-dataprovider/data/de/training-datasets/v4/features.out.json/"
out_dir = "/ML_DATA/gfk/DE/"
filename = "part-r-00000-93628840-fd71-4a78-8bdb-6cafdf2b2738"
inp_ext  = "json"
datafile = home + inp_dir + filename + '.' + inp_ext
# datafile = "./test_sample2.json"
print("open file : ", datafile)

# Create large array to (hopefully) fit all cookies (rows) x BehaviorIDs (columns)
array_size = (8192,32768)
array_type = np.uint32 # np.uint8 # np.uint32 # np.float32
na = np.zeros(array_size, array_type)
array_type_str = type(na[0,0]).__name__

idx_names = ['hhid','uid','cookieid']
# df = pd.DataFrame(columns=['hhid','uid','cookieid']) #, index=['hhid','uid','cookieid'])
# df = df.set_index(['hhid','uid','cookieid'])

# start import
print("start import into array of type ", array_type_str),
start = time.time()

i = 0
col_names = {} # type dictionary (unsorted !) = {'LTMBEH_1': 0, 'LTMBEH_3': 1, 'LTMBEH_2': 2}
col_idx   = 0
row_names = {} # type dictionary (unsorted !) = {('0011', '01', 'cookie_2'): 1, ('0022', '01', 'cookie_1'): 0}
row_idx   = 0

idx_names = ['hhid','uid','cookieid']
df = pd.DataFrame(columns=idx_names + ['np_idx'])
df.set_index(idx_names,inplace=True)

with open(datafile) as f:
    lines = f.readlines()
    for line in lines:
        if line[0]=='{':         # if line does not start with curly bracket then it is comment or empty > ignore
            d = json.loads(line)
            i += 1

            # get colum index from behavioID , create new column if column for behavioID does not yet exist
            bh_id = d['featurekey']     # we want to avoid key error
            if bh_id not in col_names:
                col_names[bh_id] = col_idx  # create a new dictionary entry > 'LTMBEH_xxx': c
                c = col_idx                 # now we know column index already
                col_idx += 1
            else:
                c = col_names[bh_id]        # if column exist > get column index


            cu_id = (d['hhid'],d['uid'],d['cookieid']) # tuple
            # cu_id = [d['hhid'],d['uid'],d['cookieid']] # list > ERROR unhashable

            if cu_id not in df.index:
                df.at[cu_id , 'np_idx'] = row_idx
                r = row_idx
                row_idx += 1
            else:
                r = df.at[cu_id, 'np_idx']

            if d['featurevalue'] != 0:
    #            if na[r,c] < 255: # uncomment for array_type = np.uint8
                    na[r,c] += 1
            if i % 1e6 == 0:
                print(int(i/1e6), 'million lines processed in' , (time.time() - start), "sec")

time_fit = (time.time() - start)
print(i, "lines processed")
print("DONE in ", time_fit, "sec")

print("col_idx   = ", col_idx)
print("row_idx   = ", row_idx)

# print("col_names = ", col_names)
# for i,r in enumerate(row_names):
#    print(i,r,na[i])

# remove MultiIndex, set index to cokkie colum only
df_index_cookie = df.reset_index()

df_index_cookie_shape = df_index_cookie.shape
print("before checkig and removing duplicates - df_index_cookie.shape = ", df_index_cookie_shape)

df_index_cookie.drop_duplicates(keep=False, inplace=True) # subset=['cookieid']  # remove rows which assign the same value again
df_index_cookie.drop_duplicates(subset='cookieid', keep=False, inplace=True)
if df_index_cookie.shape != df_index_cookie_shape:
    print("removed rows wih drop_duplicate cookies - df.shape = ", df_index_cookie.shape)
df_index_cookie.set_index('cookieid', inplace=True)

print("create pandas.DataFrame from numpy-array, row_names and col_names ...")
start = time.time()
# create sorted lists of row_names and col_names by index
# http://pythoncentral.io/how-to-sort-python-dictionaries-by-key-or-value/
# > Custom sorting algorithms with Python dictionaries
# df = pd.DataFrame(na[0:row_idx,0:col_idx], index=sorted(row_names, key=row_names.__getitem__), columns=sorted(col_names, key=col_names.__getitem__))
df_data = pd.DataFrame(na[0:row_idx,0:col_idx], columns=sorted(col_names, key=col_names.__getitem__))
print("DONE in ", (time.time() - start), "sec")

print("df_data.shape = ", df_data.shape);print()

print("df_data.index.has_duplicates = ", df_data.index.has_duplicates)
print("df_data.max value = ", df_data.max(axis=0).max())
# df_data.max(axis=0).hist(bins=100)

# print("generating statistics ...")
# print(df_data.describe(include='all'))

# max number of PCA components = nuber of features/colums
n_components_pca = min(col_idx,64)
print("start PCA with n_components =", n_components_pca)
start = time.time()
pca = PCA(n_components=n_components_pca)
pca.fit(na[0:row_idx,0:col_idx])
print("DONE in ", (time.time() - start), "sec")
pca_evr = pca.explained_variance_ratio_
print("pca.explained_variance_ratio_ = ")
print(pca_evr)
ps = pd.Series(pca_evr)
# ps.plot()

print("write pandas.DataFrame as picle file ...")
start = time.time()
df_data.to_pickle(home + out_dir + filename + '_' + array_type_str + time.strftime("_%Y-%m-%d_%H-%M-%S") + array_type_str + '.pkl')
print("DONE in ", (time.time() - start), "sec")
# df = pd.read_pickle(file_name)
'''
print("write pandas.DataFrame as csv file ...")
start = time.time()
df_data.to_csv(home + out_dir + filename + '_' + array_type_str + time.strftime("_%Y-%m-%d_%H-%M-%S") + '.csv', sep='\t')
print("DONE in ", (time.time() - start), "sec")
'''
# store = pd.HDFStore(filename_out + '.h5')
# store['filename'] = df  # save it
# store['df']  # load it

ps.to_csv(home + out_dir + filename + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '_pca_evr.csv', sep='\t')