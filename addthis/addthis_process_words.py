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
import operator
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
print("number of unique cookies = ", df['addthis_ID'].unique().shape[0])
print()

lang = 'de'
print("loading spacy vocab:", lang)
start = time.time()
nlp = spacy.load(lang)
print("done in %.2f seconds)" % (time.time() - start))

print('-----------------------------------------------------------------')
topics  = list(['Wein', 'Computer', 'Reisen', 'fliegen', 'Gesundheit', 'Sport', 'Finanzen', 'Geld', 'kochen'])
samples = list([9, 12, 27, 121])

for sample in samples:
    words = df.iat[sample,-1]
    print(words, '\n')
    doc = nlp(words)
    for w in doc:
        print(w.text, w.pos_ , ' | ', end='')
    print()
#   print("doc.vector = ", doc.vector)  
    sim_list = list()
    for t in topics:
        sim_list.append( [t, doc.similarity(nlp(t)) ] )
    print(pd.DataFrame(sorted(sim_list, key = operator.itemgetter(1), reverse=True)))
    print('-----------------------------------------------------------------')

words = 'Urlaub Flugzeug Hotel Strand Ferien'
print(words, '\n')
doc = nlp(words)
for w in doc:
    print(w.text, w.pos_ , ' | ', end='')
print()
sim_list = list()
for t in topics:
    sim_list.append( [t, doc.similarity(nlp(t)) ] )
print(pd.DataFrame(sorted(sim_list, key = operator.itemgetter(1), reverse=True)))
print('-----------------------------------------------------------------')

