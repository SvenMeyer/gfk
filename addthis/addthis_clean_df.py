'''
read addthis raw text log file into pandas DataFrame

df.columns :  Index(['TIMESTAMP', 'UID', 'GEO', 'URL', 'CATEGORIES', 'USERAGENT',
       'META_KEYWORDS', 'KEY_TERMS', 'ENTITIES']
'''

import os
import glob
import time
import csv
import pandas as pd
from pandas import DatetimeIndex
import numpy as np
import re


WORK_DIR = "/media/sumeyer/SSD_2/ML_DATA/GFK/AddThis/data_panel"
INPUT_FILE_NAME = "addthis_panel.tsv"

df = pd.read_csv(os.path.join(WORK_DIR, INPUT_FILE_NAME), sep='\t', skipinitialspace=True, nrows=None, memory_map=True,
                 dtype={'TIMESTAMP':np.uint64, 'UID':str, 'GEO':str, 'URL':str, 'CATEGORIES':str, 
                        'USERAGENT':str, 'META_KEYWORDS':str, 'KEY_TERMS':str, 'ENTITIES':str},
                 error_bad_lines=False, warn_bad_lines=True, quoting=csv.QUOTE_NONE) # index_col='UID',
print("df.shape = ", df.shape)
print("df.columns : ", df.columns)
df.rename(columns={'UID':'addthis_ID'}, inplace=True)
df.sort_values(['addthis_ID', 'TIMESTAMP'], ascending=[True, True], inplace=True)
df.set_index(['addthis_ID'], inplace=True)
df.fillna('', inplace=True)

word_cols = list(['CATEGORIES','META_KEYWORDS','KEY_TERMS','ENTITIES'])

# clean word colums, remove special chars and join all words into one additional 'word'-column
# http://stackoverflow.com/questions/26886653/pandas-create-new-column-based-on-values-from-other-columns
def clean_words(row):
    rc = re.compile(r'(\%\w\w)|[^\w]', re.UNICODE)
    words = rc.sub(" ", row['CATEGORIES']) + ' ' + rc.sub(" ", row['META_KEYWORDS']) + ' ' + rc.sub(" ", row['KEY_TERMS']) + ' ' + rc.sub(" ", row['ENTITIES'])
    return words

df['words'] = df.apply(lambda row: clean_words(row), axis=1)
    
'''
# this creates a SettingWithCopyWarning
rc = re.compile(r'(\%\w\w)|[^\w]', re.UNICODE)
df['words'] = df.apply(lambda row: rc.sub(" ", row['CATEGORIES']) + ' ' +
                                 rc.sub(" ", row['META_KEYWORDS']) + ' ' +
                                 rc.sub(" ", row['KEY_TERMS']) + ' ' +
                                 rc.sub(" ", row['ENTITIES']) , axis=1)
'''

print("df.columns = ", df.columns)
print("df.index   = ", df.index.names)
print("df.shape   = ", df.shape)
print("number of events = ", df.shape[0])
print("unique cookies   = ", len(df.index.unique()))
print("average events / cookie = ", df.shape[0] / len(df.index.unique() ))
print(df.info())

print("writing csv file")
df.drop(word_cols, inplace=True, axis=1)
df.to_csv(os.path.join(WORK_DIR, 'addthis_panel_words.tsv'), sep='\t')
df[:1000].to_csv(os.path.join(WORK_DIR, 'addthis_panel_words_sample_1k.tsv'), sep='\t')
print("DONE")