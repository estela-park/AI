############################################################################
# n 일 ~ n + 19 일의 20일의 데이터를 이용, n + 20, n + 21 일의 종가를 예측 #
############################################################################

# When data is resonably simplistic, and the amount is moderate, 
# Complex and multi-layered model rather show compromised performance.
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split

def split_x(dataset, size):
    lst = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        lst.append(subset)
    return np.array(lst)   

dataset_samsung = pd.read_csv('./samsung/_data/samsung.csv', encoding='EUC-KR')
dataset_samsung = dataset_samsung.iloc[:2601, [0, 1, 2, 3, 4, 10]]
dataset_samsung = dataset_samsung.sort_values('일자', ascending=True).set_index('일자')

x1 = np.array(dataset_samsung.iloc[:2589, [0, 1, 2, 4]])
x1 = split_x(x1, 20) 

y = np.array(dataset_samsung.iloc[20:2591, 3])
y = split_x(y, 2)

x1_predict = np.array(dataset_samsung.iloc[2581:, [0, 1, 2, 4]])
x1_predict = split_x(x1_predict, 20)

dataset_sk = pd.read_csv('./samsung/_data/SK.csv', encoding='EUC-KR')
dataset_sk = dataset_sk.iloc[:2601, [0, 1, 2, 3, 4, 10]]
dataset_sk = dataset_sk.sort_values('일자', ascending=True).set_index('일자')

x2 = np.array(dataset_sk.iloc[:2589, [0, 1, 2, 4]])
x2 = split_x(x2, 20)

x2_predict = np.array(dataset_sk.iloc[2581:, [0, 1, 2, 4]])
x2_predict = split_x(x2_predict, 20)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=200, random_state=72)

x1_train = x1_train.reshape(2370, 20*4)
x2_train = x2_train.reshape(2370, 20*4)
x1_predict = x1_predict.reshape(1, 80)
x2_predict = x2_predict.reshape(1, 80)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x1_train)
x1_predict = min_max_scaler.transform(x1_predict)
min_max_scaler.fit(x2_train)
x2_predict = min_max_scaler.transform(x2_predict)

x1_predict = x1_predict.reshape(1, 20, 4)
x2_predict = x2_predict.reshape(1, 20, 4)

model = load_model('./samsung/_save/PJH_80000_W.hdf5')

print('2021-07-23 삼성전자 종가 예측:',model.predict([x1_predict, x2_predict])[0][1])
