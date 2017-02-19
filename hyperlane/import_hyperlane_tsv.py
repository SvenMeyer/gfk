'''

import hyperlane tsv file

input format:
ignore          hhid+uid  col_index   value
1482105600      00005601        25      1
1482105600      00005601        35      0
1482105600      00005601        45      0
1482105600      00005601        57      0
1482105600      00005601        82      0
1482105600      00005601        92      2
1482105600      00005601        102     1


'''
import os
import time
import json
import numpy as np
import pandas as pd

# OPEN FILE
home = os.path.expanduser("~")
dir  = "/ML_DATA/gfk/DE/hyperlane/"
filename_inp = "targetgroup_attributes_DE_rev2"
file_ext_inp = ".tsv"
file_inp = home + dir + filename_inp + file_ext_inp
print("open file : ", file_inp)

# start import
print("start import..."),
start = time.time()

if file_ext_inp == ".tsv":
    sep_str='\t'
else:
    sep_str=','

# df = pd.read_csv(file_inp, sep=sep_str, header=None) # OK
# df = pd.read_csv(file_inp, sep=sep_str, names=['constant','hhid-uid','col_index','value']) # OK
df = pd.read_csv(file_inp, sep=sep_str, names=['constant','hhid-uid','col_index','value'], index_col='hhid-uid')


time_fit = (time.time() - start)
print("DONE in ", time_fit, "sec")

# print first 10 rows
print(df[0:10])


