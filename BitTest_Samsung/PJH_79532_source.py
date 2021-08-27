# 모델 소스코드
#################################################################################
# n 일 ~ n + 19 일의 20일의 데이터를 이용, n + 20, n + 21, n + 22 일의 시가를 예측 #
#################################################################################
import datetime
import numpy as np
import pandas as pd
import time
from tensorflow.keras.layers import Conv1D, LSTM, Dropout, Dense, Input, concatenate, MaxPool1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def split_x(dataset, size):
    lst = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        lst.append(subset)
    return np.array(lst)   

dataset_samsung = pd.read_csv('./samsung/_data/samsung.csv', encoding='EUC-KR')
# 원자료: 시간역순, 일자: 2011/1/3~ 시가, 종가, 고가, 저가, 거래량 자르기
dataset_samsung = dataset_samsung.iloc[:2601, [0, 1, 2, 3, 4, 10]]
# 시간순으로 재배열, 시간을 index로 사용
dataset_samsung = dataset_samsung.sort_values('일자', ascending=True).set_index('일자')

# 훈련데이터로 사용할 부분, loss 측정, R2 측정
x1 = np.array(dataset_samsung.iloc[:2588, 1:])
x1 = split_x(x1, 20) 
y = np.array(dataset_samsung.iloc[20:2591, 0])
y = split_x(y, 3)

# Performance index
x1_predict = np.array(dataset_samsung.iloc[2569:2598, 1:])
x1_predict = split_x(x1_predict, 20)

y_actual = np.array(dataset_samsung.iloc[2589:, 0])
y_actual = split_x(y_actual, 3)

dataset_sk = pd.read_csv('./samsung/_data/SK.csv', encoding='EUC-KR')
dataset_sk = dataset_sk.iloc[:2601, [0, 1, 2, 3, 4, 10]]
dataset_sk = dataset_sk.sort_values('일자', ascending=True).set_index('일자')

# 훈련데이터로 사용할 부분, loss 측정, R2 측정
x2 = np.array(dataset_sk.iloc[:2588, 1:])
x2 = split_x(x2, 20)

# Performance index
x2_predict = np.array(dataset_sk.iloc[2569:2598, 1:])
x2_predict = split_x(x2_predict, 20)

# 훈련용 데이터를 train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=200, random_state=99)

# Scaling
x1_train = x1_train.reshape(2369, 20*4)
x1_test = x1_test.reshape(200, 20*4)
x1_predict = x1_predict.reshape(10, 20*4)
x2_train = x2_train.reshape(2369, 20*4)
x2_test = x2_test.reshape(200, 20*4)
x2_predict = x2_predict.reshape(10, 20*4)

min_max_scaler = MinMaxScaler()
x1_train_mm = min_max_scaler.fit_transform(x1_train)
x1_test_mm = min_max_scaler.transform(x1_test)
x1_predict_mm = min_max_scaler.transform(x1_predict)
x2_train_mm = min_max_scaler.fit_transform(x2_train)
x2_test_mm = min_max_scaler.transform(x2_test)
x2_predict_mm = min_max_scaler.transform(x2_predict)

x1_train_mm = x1_train_mm.reshape(2369, 20, 4)
x1_test_mm = x1_test_mm.reshape(200, 20, 4)
x1_predict_mm = x1_predict_mm.reshape(10, 20, 4)
x2_train_mm = x2_train_mm.reshape(2369, 20, 4)
x2_test_mm = x2_test_mm.reshape(200, 20, 4)
x2_predict_mm = x2_predict_mm.reshape(10, 20, 4)

# Modelling
input_ss = Input(shape=(20, 4))
hl_ss = Conv1D(196, 2, activation='relu')(input_ss)
hl_ss = Dropout(0.2)(hl_ss)
hl_ss = Conv1D(98, 2, activation='relu')(hl_ss)
hl_ss = MaxPool1D()(hl_ss)

input_sk = Input(shape=(20, 4))
hl_sk = Conv1D(196, 2, activation='relu')(input_sk)
hl_sk = Dropout(0.2)(hl_sk)
hl_sk = Conv1D(98, 2, activation='relu')(hl_sk)
hl_sk = MaxPool1D()(hl_sk)

outL = concatenate([hl_ss, hl_sk])
outL = Conv1D(49, 2, activation='relu')(outL)
outL = GlobalAveragePooling1D()(outL)
outL = Dense(3)(outL)

model = Model(inputs=[input_ss, input_sk], outputs=[outL])

# Compilation & Fitting
# loss: mse, optimizer: adam, batch: 16, validation: 0.25, accurach: R2
model.compile(loss='mse', optimizer='adam')

date = datetime.datetime.now()
date_time = date.strftime('%m%d-%H%M')
f_path = f'./samsung/_save2/test_{date_time}' + '_{epoch:04d}_{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', patience=36, mode='auto', verbose=2, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=0, save_best_only=True, filepath=f_path)

start = time.time()
model.fit([x1_train_mm, x2_train_mm], y_train, batch_size=16, epochs=360, verbose=0, callbacks=[es, mcp], validation_split=0.25, shuffle=False)
end = time.time() - start

print('it took', end/60, 'minutes and', end%60,'seconds')
print('now is',date.strftime('%m%d-%H%M'))

loss = model.evaluate([x1_test_mm, x2_test_mm], y_test)

print('loss:',loss)