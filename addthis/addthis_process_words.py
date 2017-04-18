"""
read addthis raw text log file and generate doc-vector for each event

1) INPUT :

one RAW AddThis text logfile with cleaned words (tab-separated CSV-file)
columns ['TIMESTAMP', 'UID', 'GEO', 'URL', 'CATEGORIES', 'USERAGENT',
         'META_KEYWORDS', 'KEY_TERMS', 'ENTITIES', 'words']

2 OUTPUT :

"""
TEST = True

import os
# import glob
import time
# import csv
import pandas as pd
import numpy as np
# import re
import spacy

# asuming we are on the laptop VM
HOME_DIR = "/media/sf_SHARE"
if not os.path.isdir(HOME_DIR): # we are onthe PC
    HOME_DIR = "/media/sumeyer/SSD_2"
WORK_DIR = os.path.join(HOME_DIR, "ML_DATA/GFK/AddThis/data_panel")
INPUT_FILE_NAME = "addthis_panel_words.tsv" if not TEST else "addthis_panel_words_sample_1k.tsv"

input_file_path = os.path.join(WORK_DIR, INPUT_FILE_NAME)
print("loading file : ", input_file_path)
df = pd.read_csv(input_file_path, sep='\t', dtype=str)
print("df.shape = ", df.shape)
print("df.columns : ", df.columns)

print("loading spacy vocab")
spacy.load('en')


