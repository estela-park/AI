import numpy as np
import pandas as pd

def split_x(dataset, size):
    lst = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        lst.append(subset)
    return np.array(lst)   

# check the location of this .py file
# import os
# print(os.getcwd())

# File is not a recognized excel file: the error arose because files were forcibly saved as .xls files.
# > .xls to .csv
# When CSV file have variable number of columns, read_csv infers the number of columns from the first few rows
# > forcibly gave column names
# Most of the csv CSV are created using delimiter of '/t'
# > sep='\t'
# When data type is mixed, parsing would not be possible
# > error_bad_lines=False

# import chardet
# rawdata = open('./samsung/_data/samsung.csv', 'rb').read()
# result = chardet.detect(rawdata)
# charenc = result['encoding']
# >> None: encoding is messed up I quess

dataset_samsung = pd.read_csv('./samsung/_data/samsung.csv', encoding='EUC-KR')
# .keys() returns index for Series, columns for DataFrame

print(dataset_samsung.keys())
print(dataset_samsung.head(15))
print(type(dataset_samsung))

print(dataset_samsung[dataset_samsung['일자'] == '2011/01/03'].index.tolist())
# 2600

# Trimming by (Date, columns)
dataset_samsung = dataset_samsung.iloc[:2601, [0, 1, 2, 3, 4, 10]]
print(dataset_samsung.keys())
print(dataset_samsung.head(15))
print(dataset_samsung.tail(5))

# Arranging timely

# dataset_samsung = dataset_samsung.sort_values('일자', ascending=True)
# > index is still opposite
dataset_samsung = dataset_samsung.sort_values('일자', ascending=True).set_index('일자')
print(dataset_samsung.keys())
print(dataset_samsung.head(15))
print(dataset_samsung.tail(5))

# Practice makes it perfect
data_pract = dataset_samsung.iloc[:10, :]

'''
2011/01/03  19100.0  19320.0  19000.0  19160.0  13278100.0
2011/01/04  19120.0  19220.0  18980.0  19160.0  13724400.0
2011/01/05  19100.0  19100.0  18840.0  18840.0  16811200.0
2011/01/06  18840.0  18980.0  18460.0  18600.0  19374400.0
2011/01/07  18300.0  18580.0  18280.0  18420.0  23172350.0
2011/01/10  18360.0  18560.0  18180.0  18340.0  18355350.0
2011/01/11  18340.0  18440.0  18160.0  18260.0  19562100.0
2011/01/12  18280.0  18660.0  18280.0  18600.0  20605350.0
2011/01/13  18980.0  18980.0  18340.0  18440.0  28404150.0
2011/01/14  18560.0  18680.0  18300.0  18660.0  15384700.0
'''

# Input-Output parting
x_data = np.array(data_pract.iloc[:, [0, 1, 2, 4]])
print(x_data)
print(type(x_data))
print(x_data.shape)

x_data = split_x(x_data, 7)
print(x_data)
print('check here**************************')
print(x_data.shape)
# predict: 3, 7, 4/ actual: 4, 7, 4

y_data = np.array(data_pract.iloc[:, 3])
print(y_data)
print(type(y_data))

y_data = split_x(y_data, 2)
print(y_data)
print(y_data.shape)

# Shaping
x_data = x_data[:2]
y_data = y_data[7:]
print(x_data)
print(x_data.shape)
print(y_data)
print(y_data.shape)

# Go big!
print('_'*75)
big_pract = dataset_samsung.iloc[:30, :]

big_x = np.array(big_pract.iloc[:, [0, 1, 2, 4]])
big_x = split_x(big_x, 7)

big_y = np.array(big_pract.iloc[:, 3])
big_y = split_x(big_y, 2)

big_x = big_x[:22]
big_y = big_y[7:]

'''
**DataSetUp

import numpy as np
import pandas as pd
import os

def split_x(dataset, size):
    lst = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        lst.append(subset)
    return np.array(lst)   

dataset_samsung = pd.read_csv('./samsung/_data/samsung.csv', encoding='EUC-KR')
dataset_samsung = dataset_samsung.iloc[:2601, [0, 1, 2, 3, 4, 10]]
dataset_samsung = dataset_samsung.sort_values('일자', ascending=True).set_index('일자')

data = dataset_samsung.iloc[:N, :]

x = np.array(data.iloc[:, [0, 1, 2, 4]])
x = split_x(x, 7)

y = np.array(data.iloc[:, 3])
y = split_x(y, 2)

x = x[:N-8]
y = y[7:]

'''