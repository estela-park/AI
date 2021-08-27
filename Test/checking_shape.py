import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling1D, GlobalAvgPool1D, Dropout, Dense, GRU, Input, Conv1D, concatenate, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def split_x(dataset, size):
    lst = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        lst.append(subset)
    return np.array(lst)   

dataset_samsung = pd.read_csv('./samsung/_data/samsung.csv', encoding='EUC-KR')
dataset_samsung = dataset_samsung.iloc[:2601, [0, 1, 2, 3, 4, 10]]
dataset_samsung = dataset_samsung.sort_values('일자', ascending=True).set_index('일자')
# (2601, 5)

x1_predict = np.array(dataset_samsung.iloc[2568:2599, [0, 1, 2, 4]])
x1_predict = split_x(x1_predict, 20)
# (14, 7, 4)

y_actual = np.array(dataset_samsung.iloc[2590:, 3])
y_actual = split_x(y_actual, 2)
# (14, 2)

x1 = np.array(dataset_samsung.iloc[:2589, [0, 1, 2, 4]])
x1 = split_x(x1, 20)
# (2579, 7, 4)  

y = np.array(dataset_samsung.iloc[20:2591, 3])
y = split_x(y, 2)
# (2579, 2)

dataset_sk = pd.read_csv('./samsung/_data/SK.csv', encoding='EUC-KR')
dataset_sk = dataset_sk.iloc[:2601, [0, 1, 2, 3, 4, 10]]
dataset_sk = dataset_sk.sort_values('일자', ascending=True).set_index('일자')
# (2601, 5)

x2_predict = np.array(dataset_sk.iloc[2568:2599, [0, 1, 2, 4]])
x2_predict = split_x(x2_predict, 20)
# (14, 7, 4)

x2 = np.array(dataset_sk.iloc[:2589, [0, 1, 2, 4]])
x2 = split_x(x2, 20)
# (2579, 7, 4)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=200, random_state=72)

print(x1_train.shape)
print(x1_test.shape)
print(x1_predict.shape)