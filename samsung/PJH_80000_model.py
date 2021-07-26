############################################################################
# n 일 ~ n + 19 일의 20일의 데이터를 이용, n + 20, n + 21 일의 종가를 예측 #
############################################################################

import numpy as np
import pandas as pd
import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, Input, concatenate, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def split_x(dataset, size):
    lst = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        lst.append(subset)
    return np.array(lst)   

# Data-Prep
dataset_samsung = pd.read_csv('./samsung/_data/samsung.csv', encoding='EUC-KR')
dataset_samsung = dataset_samsung.iloc[:2601, [0, 1, 2, 3, 4, 10]]
dataset_samsung = dataset_samsung.sort_values('일자', ascending=True).set_index('일자')

x1_predict = np.array(dataset_samsung.iloc[2561:2590, [0, 1, 2, 4]])
x1_predict = split_x(x1_predict, 20)

y_actual = np.array(dataset_samsung.iloc[2590:, 3])
y_actual = split_x(y_actual, 2)

x1 = np.array(dataset_samsung.iloc[:2589, [0, 1, 2, 4]])
x1 = split_x(x1, 20) 

y = np.array(dataset_samsung.iloc[20:2591, 3])
y = split_x(y, 2)

dataset_sk = pd.read_csv('./samsung/_data/SK.csv', encoding='EUC-KR')
dataset_sk = dataset_sk.iloc[:2601, [0, 1, 2, 3, 4, 10]]
dataset_sk = dataset_sk.sort_values('일자', ascending=True).set_index('일자')

x2_predict = np.array(dataset_sk.iloc[2561:2590, [0, 1, 2, 4]])
x2_predict = split_x(x2_predict, 20)

x2 = np.array(dataset_sk.iloc[:2589, [0, 1, 2, 4]])
x2 = split_x(x2, 20)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=200, random_state=72)

# Scaling
x1_train = x1_train.reshape(2370, 20*4)
x1_test = x1_test.reshape(200, 20*4)
x1_predict = x1_predict.reshape(10, 20*4)
x2_train = x2_train.reshape(2370, 20*4)
x2_test = x2_test.reshape(200, 20*4)
x2_predict = x2_predict.reshape(10, 20*4)

min_max_scaler = MinMaxScaler()
x1_train_mm = min_max_scaler.fit_transform(x1_train)
x1_test_mm = min_max_scaler.transform(x1_test)
x1_predict_mm = min_max_scaler.transform(x1_predict)
x2_train_mm = min_max_scaler.fit_transform(x2_train)
x2_test_mm = min_max_scaler.transform(x2_test)
x2_predict_mm = min_max_scaler.transform(x2_predict)

x1_train_mm = x1_train_mm.reshape(2370, 20, 4)
x1_test_mm = x1_test_mm.reshape(200, 20, 4)
x1_predict_mm = x1_predict_mm.reshape(10, 20, 4)
x2_train_mm = x2_train_mm.reshape(2370, 20, 4)
x2_test_mm = x2_test_mm.reshape(200, 20, 4)
x2_predict_mm = x2_predict_mm.reshape(10, 20, 4)

# Modelling
input_ss = Input(shape=(20, 4))
hl_ss = LSTM(units=196, activation='relu', dropout=0.2)(input_ss)
hl_ss = Dropout(0.35)(hl_ss)
hl_ss = Dense(98, activation='relu')(hl_ss)
hl_ss = Dropout(0.35)(hl_ss)

input_sk = Input(shape=(20, 4))
hl_sk = LSTM(units=196, activation='relu', dropout=0.2)(input_sk)
hl_sk = Dropout(0.2)(hl_sk)
hl_sk = Dense(98, activation='relu')(hl_sk)
hl_sk = Dropout(0.2)(hl_sk)

outL = concatenate([hl_ss, hl_sk])
outL = Dense(49, activation='relu')(outL)
outL = Dropout(0.25)(outL)
outL = Dense(14, activation='relu')(outL)
outL = Dense(2)(outL)

model = Model(inputs=[input_ss, input_sk], outputs=[outL])

# Compilation & Training

model.compile(loss='mae', optimizer='adam')
model.save('./samsung/_save/model_save.h5')
import datetime
date = datetime.datetime.now()
date_time = date.strftime('%m%d-%H%M')
f_path = f'./samsung/_save/test_{date_time}' + '_{epoch:04d}_{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', patience=32, mode='auto', verbose=2, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=2, filepath=f_path)

start = time.time()
model.fit([x1_train_mm   , x2_train_mm], y_train, batch_size=16, epochs=200, verbose=2, callbacks=[es, mcp], validation_split=0.15, shuffle=False)
model.save_weights(filepath=f_path)
end = time.time() - start

#Evaluation
print('----------------- robust ----------------')
loss = model.evaluate([x1_test_mm, x2_test_mm], y_test)
predict = model.predict([x1_predict_mm, x2_predict_mm])
r2 = r2_score(y_actual, predict)

print('it took', end/60, 'minutes and', end%60,'seconds')
print('loss:',loss,'R2:',r2)
print('prediction for',y_actual,'is',predict)

'''
----------------- minmax ----------------Epoch 00043
it took 8 minutes and 22 seconds
loss: 3760.004638671875 R2: -134.16825231245312
prediction for [80800. 79900. 79400.] is [77795.625 70043.664 77929.945]
----------------- minmax ----------------Epoch 00037
it took 7 minutes and 1 second
loss: 2780.99365234375 R2: -74.25961504992847
prediction for [80800. 79900 79400.] is [83413.72  83286.33 84281.914]


----------------- maxabs ----------------Epoch 00040
it took 7 minutes and 52 seconds
loss: 4408.0654296875 R2: -34.397311456991844
prediction for [80800. 79900. 79400.] is [74337.266 77272.57 77561.79]
----------------- maxabs ----------------Epoch 00034
it took 6 minutes and 23 seconds
loss: 3546.06396484375 R2: -6.10385353770152
prediction for [80800. 79900 79400.] is [78643.11  79956.234 80114.67]


----------------- robust ----------------Epoch 00046
it took 8 minutes and 50 seconds
loss: 3161.72998046875 R2: -14.945589299063403
prediction for [80800. 79900. 79400.] is [79099.414 76563.625 77454.]
: early stopping
----------------- robust ----------------Epoch 00046
it took 8 minutes and 55 seconds
loss: 2646.7998046875 R2: -25.276212709811272
prediction for [80800. 79900. 79400.] is [75466.336 80388.234]
'''