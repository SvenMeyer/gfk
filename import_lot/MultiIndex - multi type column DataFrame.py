# pandas DataFrame create,access,append MultiIndex with different Column types - SQL table style

'''
I thought by using the huge pandas.DataFrame library it should be pretty straight forward to do all the standard stuff you can do with an SQL table .. but after looking into many options I still haven't found a good working solution.

Requirements:

    table with a 4 columns with different data types (uint32, string,...) , 3 off them should work as index
    many (>10k) additional columns of type int8
    initially I had the idea to add rows and columns dynamically, but that turned out to be very slow (using df.at[row, col] = y)
    I ended up creating a DataFrame with a few columns with different types and join it with another large DataFrame created from a numpy array with elements of type uint8

    ... that looked quite good, but now nothing works to access, add or set array elements using the index

    http://stackoverflow.com/questions/42320736/pandas-dataframe-create-access-append-multiindex-with-different-column-types-s
'''

# STATUS . DOES NOT WORK
# 1) concat > raise KeyError('%s not in index' % objarr[mask])
#             KeyError: "['000001' '01' 'abcdef'] not in index"
# 2) join   >  a) new rows and colums contain NaNs, so I can't increment the value by +=1
#                (without a lenghty "if NaN then .. else ... " (what I wanted to avoid for performnce reasons)
#              b) I lost the type of the data elements , they are not uint8 anymore (.. and I have millions of them



import numpy as np
import pandas as pd

# create DataFrame

idx_names = ['hhid','uid','cookieid']
col_names = ['y']
df = pd.DataFrame(columns = idx_names + col_names)

# create DataFrame from numpy array

npa = np.zeros((5,10),dtype=np.uint8)
dfa = pd.DataFrame(npa)

# add DataFrames column-wise

# t = pd.concat([df,dfa], axis=1)
t = df.join(pd.DataFrame(dfa, index=df.index))

# set index columns

t.set_index(idx_names,inplace=True)

print(t)

'''
>>> t
                   y  0  1  2  3  4  5  6  7  8  9
    A   B   C
    NaN NaN NaN  NaN  0  0  0  0  0  0  0  0  0  0
            NaN  NaN  0  0  0  0  0  0  0  0  0  0
            NaN  NaN  0  0  0  0  0  0  0  0  0  0
            NaN  NaN  0  0  0  0  0  0  0  0  0  0
            NaN  NaN  0  0  0  0  0  0  0  0  0  0
'''

# get index of column

i0 = t.columns.get_loc('y')
i1 = t.columns.get_loc(0)

t.at[ ('000001','01','abcdef') , 'y' ] = 'M'
t.at[ ('000001','01','abcdef') ,  0  ] =  1
t.at[ ('000001','02','xyzxyz') , 'y' ] = 'F'
t.at[ ('000001','02','xyzxyz') ,  9  ] =  1

print(t)

