import os
import glob
import pandas as pd
import numpy as np

TEST = True
COL_NAME_PID ="pnr"

HOME_DIR = os.path.expanduser("~")
LOT_DIR  = "/ML_DATA/GFK/DE/Lotame/"
LOT_FILE = "part-r-00000-93628840-fd71-4a78-8bdb-6cafdf2b2738"
GXL_DIR  = "/ML_DATA/GFK/DE/Hyperlane/unimputed-target-groups/2017-02-01/"
GXL_FILE = "GXL_data.tsv"
TARGET   = "dep_tg_bin_gender_1"

if TEST:
    LOT_FILE = "Lot_data_test"
    GXL_FILE = "GXL_data_test.tsv"

df_GXL = pd.read_csv(HOME_DIR + GXL_DIR + GXL_FILE, sep='\t', dtype=np.int32)
print("df_GXL.shape = ", df_GXL.shape)
print(df_GXL.iloc[:10,:5])

#df_Lot = pd.read_pickle(HOME_DIR + LOT_DIR + LOT_FILE + ".pkl")
df_Lot = pd.read_csv(HOME_DIR + LOT_DIR + LOT_FILE + ".tsv", sep='\t') #, dtype=np.int32)
print("df_Lot.shape = ", df_Lot.shape)
print(df_Lot.iloc[:10,:5])
'''
# Pre-Processing
# print("events in Lot table = ", df_Lot.sum(numeric_only=True).sum()) # need to exclude pnr
print("histogram : no of LOTBEH with 0 .. 99 panel visitor")
el = df_Lot.astype(bool).sum(axis=0)      # count no of non-zero entries for each LOTBEH
el[el<100].hist(bins=100, figsize=(10,10))
# drop LOTBET columns with very few entries (panel members)
MIN_panel_per_LOTBEH = 4
df_Lot.drop(el.index[el < MIN_panel_per_LOTBEH], axis=1, inplace=True)
print("removed LOTBEH columns with less than",  MIN_panel_per_LOTBEH, "entries (panel members) - df_Lot.shape = ", df_Lot.shape)

ep = df_Lot.astype(bool).sum(axis=1)
ep[ep<100].hist(bins=100, figsize=(10,10))
'''

df = pd.merge(df_Lot, df_GXL[[COL_NAME_PID, TARGET]], on=COL_NAME_PID, how='inner')
print(df.shape)
# print(df)

# extract features and target into numpy ndarry
X = df.iloc[:,2:-1].values.astype('uint8')
Y = df.iloc[:,-1].values.astype('uint8')
           