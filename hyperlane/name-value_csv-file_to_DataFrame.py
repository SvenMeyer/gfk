'''

importdf"name (column-index) x value" tabular tsv file

input format:
ignore          hhid+uid  col_index   value
1482105600      00005601        25      1
1482105600      00005601        35      0
1482105600      00005601        45      0
1482105600      00005601        57      0
1482105600      00005601        82      0
1482105600      00005601        92      2
1482105600      00005601        102     1

http://stackoverflow.com/questions/42327346/read-csv-with-column-name-x-value-pairs

df = df.drop_duplicates(keep='last')

df.shape
Out[140]: (2606244, 3)

df = df.drop_duplicates(subset=['hhid-uid','col_index'], keep='last')

df.shape
Out[142]: (2433941, 3)

'''
import os
import time
import pandas as pd

TEST=True

# OPEN FILE
home = os.path.expanduser("~")
dir  = home + "/ML_DATA/gfk/DE/hyperlane/"
filename_inp = "./targetgroup_attributes_DE_rev2_sample100"
file_ext_inp = ".tsv"


if TEST==True:
    dir = "./"
    file_inp = "./targetgroup_attributes_DE_rev2_sample100"
    # file_inp = "test"
    # file_inp = "test_duplicate"
    file_ext_inp = ".tsv"

file_inp = dir + filename_inp + file_ext_inp
print("open file : ", file_inp)

# start import
print("start import...", end=''),
start = time.time()

if file_ext_inp == ".tsv":
    sep_str='\t'
else:
    sep_str=','

# df = pd.read_csv(file_inp, sep=sep_str, header=None) # OK
df = pd.read_csv(file_inp, sep=sep_str, names=['constant','hhid-uid','col_index','value']) # OK
# df = pd.read_csv(file_inp, sep=sep_str, names=['constant','hhid-uid','col_index','value'], index_col='hhid-uid')

time_fit = (time.time() - start)
print("DONE in ", time_fit, "sec")
print("read file into DataFrame - df.shape = ", df.shape)

print("start convert DataFrame...")
start = time.time()

# drop 'constant' column
df.drop('constant', axis=1, inplace=True)
print("droped 'constant' column - df.shape = ", df.shape)

# print("unsorted ", df[0:20])
# df.sort(columns=(['hhid-uid','col_index']) )            # deprecated sort - DO NOT USE
df.sort_values(['hhid-uid','col_index'], inplace=True)  # sort optionally - for debugging
# print("sorted ", df[0:20])

df.drop_duplicates(keep='last', inplace=True) # remove rows which assign the same value again
print("removed rows which assign the same value again - df.shape = ", df.shape)
df.drop_duplicates(subset=['hhid-uid','col_index'], keep='last', inplace=True) # remove rows which assign new values (actually that should not happen)
print("removed rows which assign a new value (actually that should not happen) - df.shape = ", df.shape)

df_table = df.set_index(['hhid-uid','col_index'])['value'].unstack(fill_value=0)
# alternative option : average duplicates instead of just keeping last one
# df = df.groupby(['hhid-uid','col_index'])['value'].mean().unstack(fill_value=0)

time_fit = (time.time() - start)
print("DONE in ", time_fit, "sec");print()

print(df_table[0:10])
print("df_Lot.shape = ", df_table.shape);print()

print("write pandas.DataFrame as csv file ...", end='')
start = time.time()
df_table.to_csv(dir + filename_inp + time.strftime("_%Y-%m-%d_%H-%M-%S") + '.csv', sep='\t')
print("DONE in ", (time.time() - start), "sec")
