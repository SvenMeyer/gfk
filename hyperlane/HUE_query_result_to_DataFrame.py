'''
HUE_query_result_to_DataFrame.py
@author : Sven Meyer

example input file:

person_id,attribute_id,attribute_value_id,slice_year,slice_month,slice_start,v_loading_time
00005601,212,0,2017,01,2017-01-01,2017-02-16T15-10-41-175Z
00005601,222,0,2017,01,2017-01-01,2017-02-16T15-10-41-175Z
00008501,212,0,2017,01,2017-01-01,2017-02-16T15-10-41-175Z
00008501,222,0,2017,01,2017-01-01,2017-02-16T15-10-41-175Z
00008901,212,0,2017,01,2017-01-01,2017-02-16T15-10-41-175Z

'''

import os
import time
import pandas as pd

TEST=False

# OPEN FILE
home = os.path.expanduser("~")
home = "/media/sf_SHARE"
dir  = home + "/ML_DATA/gfk/DE/Hyperlane/HUE_2017-01/"

filename = "HUE_query_result_a001-a050_2017-01"
filename = "HUE_query_result_a051-a100_2017-01"
filename = "HUE_query_result_a101-a150_2017-01"
filename = "HUE_query_result_a150-a208_2017-01"
filename = "HUE_query_result_a209-a227_2017-01_Pet-Baby"
filename = "DE_gfk_2017-01"

file_ext_inp = ".csv"

if TEST==True:
    filename += "_test"

file_inp = dir + filename + file_ext_inp
print("open file : ", file_inp)

# start import
print("start import...", end=''),
start = time.time()

if file_ext_inp == ".tsv":
    sep_str='\t'
else:
    sep_str=','

df = pd.read_csv(file_inp, sep=sep_str) # OK

time_fit = (time.time() - start)
print("DONE in ", time_fit, "sec")
print("read file into DataFrame - df.shape = ", df.shape)
print("column names = ", df.columns.values)

print("start convert DataFrame...")
start = time.time()

person_id          = df.columns.values[0]
attribute_id       = df.columns.values[1]
attribute_value_id = df.columns.values[2]

# df.sort_values(['person_id','attribute_id'], inplace=True)  # sort optionally - for debugging

df_shape = df.shape
print("before checkig and removing duplicates - df.shape = ", df_shape)

df.drop_duplicates(subset=[person_id, attribute_id, attribute_value_id], keep='last', inplace=True) # remove rows which assign the same value again
if df.shape != df_shape:
    print("removed rows which assign the same value again - df.shape = ", df.shape)
    df_shape = df.shape

df.drop_duplicates(subset=[person_id, attribute_id], keep='last', inplace=True) # remove rows which assign new values (actually that should not happen)
if df.shape != df_shape:
    print("removed rows which assign a new value (actually that should not happen) - df.shape = ", df.shape)
    df_shape = df.shape

df_table = df.set_index(['person_id','attribute_id'])['attribute_value_id'].unstack(fill_value=0)
# alternative option : average duplicates instead of just keeping last one
# df = df.groupby(['person_id','attribute_id'])['value'].mean().unstack(fill_value=0)

time_fit = (time.time() - start)
print("DONE in ", time_fit, "sec");print()

print(df_table[0:10])
print("df_table.shape = ", df_table.shape);print()

print("write pandas.DataFrame as csv file ...", end='')
start = time.time()
df_table.to_csv(dir + "out/" + filename + time.strftime("_%Y-%m-%d_%H-%M-%S") + '.csv', sep='\t')
print("DONE in ", (time.time() - start), "sec")
