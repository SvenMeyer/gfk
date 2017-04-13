import os
import glob
import time
from datetime import date, timedelta
import gzip
import json
import pandas as pd

HOME_DIR = "/media/sf_SHARE"
if not os.path.isdir(HOME_DIR):
    HOME_DIR = os.path.expanduser("~")
WORK_DIR = os.path.join(HOME_DIR, "ML_DATA/GFK/AddThis/")
input_dir = os.path.join(WORK_DIR , 'data')
print("input_dir =", input_dir)
output_dir = os.path.join(WORK_DIR , 'data_panel')
print("output_dir =", output_dir)

cookie_df  = pd.read_csv(os.path.join(WORK_DIR,'addthis_cookies_2017-04-12.csv'))
cookie_set = set(cookie_df['addthis_uid'])

stat_file_name = os.path.join(output_dir, 'statistics.tsv')
stat_file = open(stat_file_name, "w")
print('DAY\tLINES\tEVENTS', file = stat_file)

day = date(2017,3,20)

FILE_WILDCARD = "pixelview-turbo-no-porn-de." + day.strftime('%Y%m%d') + "-????.????.log.gz"
data_files = glob.glob(os.path.join(input_dir, FILE_WILDCARD))
data_files_count = len(data_files)
print(data_files_count, "files found for day", day.strftime('%Y%m%d'))

while data_files_count > 0:
    output_file_name = os.path.join(output_dir, 'addthis_panel_' + day.strftime('%Y%m%d') + '.tsv')
    output_file = open(output_file_name, "w")
    print('TIMESTAMP\tUID\tGEO\tURL\tCATEGORIES\tUSERAGENT\tMETA_KEYWORDS\tKEY_TERMS\tENTITIES', file = output_file)
    
    n_cookies = 0
    n_lines   = 0
    for datafile in data_files:
        print("start processing file : ", datafile)
        i=0
        n=0
        with gzip.open(os.path.join(input_dir, datafile), 'r') as f:
            lines = [x.decode('utf8').strip() for x in f.readlines()]
            for line in lines[1:]:
                i += 1
                cookie = line.split('\t')[1]
                if cookie in cookie_set:
                    n += 1
                    print(line, file = output_file)
        
        print(i, "lines processed - ", end='')
        print(n, "panel cookie events found")
        n_cookies += n
        n_lines   += i
        
    output_file.close()    
    print(n_cookies, "TOTAL NUMBER of panel cookie events found for day", day.strftime('%Y%m%d'))
    print(day.strftime('%Y%m%d') + '\t' + str(n_lines) + '\t' + str(n_cookies), file = stat_file)
    
    day += timedelta(days=1)
    FILE_WILDCARD = "pixelview-turbo-no-porn-de." + day.strftime('%Y%m%d') + "-????.????.log.gz"
    data_files = glob.glob(os.path.join(input_dir, FILE_WILDCARD))
    data_files_count = len(data_files)
    print(data_files_count, "files found for day", day.strftime('%Y%m%d'))

stat_file.close()
print("*** DONE ***")