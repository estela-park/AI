# 저장된 가중치로 loss, accuracy, prediction printing 
#################################################################################
# n 일 ~ n + 19 일의 20일의 데이터를 이용, n + 20, n + 21, n + 22 일의 시가를 예측 #
#################################################################################

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def split_x(dataset, size):
    lst = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        lst.append(subset)
    return np.array(lst)   

# weight파일경로 입력하고 돌려주세요.
f_w_path = ''
# csv 파일 경로+파일명 입력하고 돌려주세요.
f_samsung = ''
f_sk = ''

dataset_samsung = pd.read_csv('{f_samsung}.csv', encoding='EUC-KR')
dataset_samsung = dataset_samsung.iloc[:2601, [0, 1, 2, 3, 4, 10]]
dataset_samsung = dataset_samsung.sort_values('일자', ascending=True).set_index('일자')

x1 = np.array(dataset_samsung.iloc[:2588, 1:])
x1 = split_x(x1, 20) 
y = np.array(dataset_samsung.iloc[20:2591, 0])
y = split_x(y, 3)

# Performance index: 예측시켜보기
x1_predict = np.array(dataset_samsung.iloc[2569:2598, 1:])
x1_predict = split_x(x1_predict, 20)

# 예측: 모범답안
y_actual = np.array(dataset_samsung.iloc[2589:, 0])
y_actual = split_x(y_actual, 3)

# 6/24 ~ 7/21 20일 데이터 이용해서 7/22, 7/23, 7/26 데이터 예측에 사용
x1_realtime = np.array([dataset_samsung.iloc[2581:, 1:]])

dataset_sk = pd.read_csv('{f_sk}.csv', encoding='EUC-KR')
dataset_sk = dataset_sk.iloc[:2601, [0, 1, 2, 3, 4, 10]]
dataset_sk = dataset_sk.sort_values('일자', ascending=True).set_index('일자')

x2 = np.array(dataset_sk.iloc[:2588, 1:])
x2 = split_x(x2, 20)

# Performance index: 예측시켜보기
x2_predict = np.array(dataset_sk.iloc[2569:2598, 1:])
x2_predict = split_x(x2_predict, 20)

# 6/24 ~ 7/21 20일 데이터 이용해서 7/22, 7/23, 7/26 데이터 예측에 사용
x2_realtime = np.array([dataset_sk.iloc[2581:, 1:]])

# 훈련환경과 동일하게 scaling
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=200, random_state=99)

x1_train = x1_train.reshape(2369, 20*4)
x1_test = x1_test.reshape(200, 20*4)
x1_predict = x1_predict.reshape(10, 20*4)
x1_realtime = x1_realtime.reshape(1, 20*4)
x2_train = x2_train.reshape(2369, 20*4)
x2_test = x2_test.reshape(200, 20*4)
x2_predict = x2_predict.reshape(10, 20*4)
x2_realtime = x2_realtime.reshape(1, 20*4)

min_max_scaler = MinMaxScaler()
x1_train_mm = min_max_scaler.fit_transform(x1_train)
x1_test_mm = min_max_scaler.transform(x1_test)
x1_predict_mm = min_max_scaler.transform(x1_predict)
x1_realtime_mm = min_max_scaler.transform(x1_realtime)
x2_train_mm = min_max_scaler.fit_transform(x2_train)
x2_test_mm = min_max_scaler.transform(x2_test)
x2_predict_mm = min_max_scaler.transform(x2_predict)
x2_realtime_mm = min_max_scaler.transform(x2_realtime)

x1_train_mm = x1_train_mm.reshape(2369, 20, 4)
x1_test_mm = x1_test_mm.reshape(200, 20, 4)
x1_predict_mm = x1_predict_mm.reshape(10, 20, 4)
x1_realtime_mm = x1_realtime_mm.reshape(1, 20, 4)
x2_train_mm = x2_train_mm.reshape(2369, 20, 4)
x2_test_mm = x2_test_mm.reshape(200, 20, 4)
x2_predict_mm = x2_predict_mm.reshape(10, 20, 4)
x2_realtime_mm = x2_realtime_mm.reshape(1, 20, 4)

model = load_model(f'{f_w_path}/PJH_79532_weights.hdf5')

loss = model.evaluate([x1_test_mm, x2_test_mm], y_test)
predict = model.predict([x1_test_mm, x2_test_mm])
r2 = r2_score(y_test, predict)
mm = model.predict([x1_realtime_mm, x2_realtime_mm])

print('loss:',loss,'accuracy(R2, calculated with test data):',r2)
print('월요일(7/26) 시가:', mm[0][2],']')
print(mm)
