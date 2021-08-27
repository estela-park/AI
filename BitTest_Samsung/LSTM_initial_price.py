import datetime
import numpy as np
import pandas as pd
import time
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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

x1 = np.array(dataset_samsung.iloc[:2588, 1:])
x1 = split_x(x1, 20) 

y = np.array(dataset_samsung.iloc[20:2591, 0])
y = split_x(y, 3)

x1_predict = np.array(dataset_samsung.iloc[2569:2598, 1:])
x1_predict = split_x(x1_predict, 20)

y_actual = np.array(dataset_samsung.iloc[2589:, 0])
y_actual = split_x(y_actual, 3)

x1_realtime = np.array([dataset_samsung.iloc[2581:, 1:]])

dataset_sk = pd.read_csv('./samsung/_data/SK.csv', encoding='EUC-KR')
dataset_sk = dataset_sk.iloc[:2601, [0, 1, 2, 3, 4, 10]]
dataset_sk = dataset_sk.sort_values('일자', ascending=True).set_index('일자')

x2 = np.array(dataset_sk.iloc[:2588, 1:])
x2 = split_x(x2, 20)

x2_predict = np.array(dataset_sk.iloc[2569:2598, 1:])
x2_predict = split_x(x2_predict, 20)

x2_realtime = np.array([dataset_sk.iloc[2581:, 1:]])

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

max_abs_scaler = MaxAbsScaler()
x1_train_ma = max_abs_scaler.fit_transform(x1_train)
x1_test_ma = max_abs_scaler.transform(x1_test)
x1_predict_ma = max_abs_scaler.transform(x1_predict)
x1_realtime_ma = max_abs_scaler.transform(x1_realtime)
x2_train_ma = max_abs_scaler.fit_transform(x2_train)
x2_test_ma = max_abs_scaler.transform(x2_test)
x2_predict_ma = max_abs_scaler.transform(x2_predict)
x2_realtime_ma = max_abs_scaler.transform(x2_realtime)

x1_train_ma = x1_train_ma.reshape(2369, 20, 4)
x1_test_ma = x1_test_ma.reshape(200, 20, 4)
x1_predict_ma = x1_predict_ma.reshape(10, 20, 4)
x1_realtime_ma = x1_realtime_ma.reshape(1, 20, 4)
x2_train_ma = x2_train_ma.reshape(2369, 20, 4)
x2_test_ma = x2_test_ma.reshape(200, 20, 4)
x2_predict_ma = x2_predict_ma.reshape(10, 20, 4)
x2_realtime_ma = x2_realtime_ma.reshape(1, 20, 4)

robust_scaler = RobustScaler()
x1_train_rb = robust_scaler.fit_transform(x1_train)
x1_test_rb = robust_scaler.transform(x1_test)
x1_predict_rb = robust_scaler.transform(x1_predict)
x1_realtime_rb = robust_scaler.transform(x1_realtime)
x2_train_rb = robust_scaler.fit_transform(x2_train)
x2_test_rb = robust_scaler.transform(x2_test)
x2_predict_rb = robust_scaler.transform(x2_predict)
x2_realtime_rb = robust_scaler.transform(x2_realtime)

x1_train_rb = x1_train_rb.reshape(2369, 20, 4)
x1_test_rb = x1_test_rb.reshape(200, 20, 4)
x1_predict_rb = x1_predict_rb.reshape(10, 20, 4)
x1_realtime_rb = x1_realtime_rb.reshape(1, 20, 4)
x2_train_rb = x2_train_rb.reshape(2369, 20, 4)
x2_test_rb = x2_test_rb.reshape(200, 20, 4)
x2_predict_rb = x2_predict_rb.reshape(10, 20, 4)
x2_realtime_rb = x2_realtime_rb.reshape(1, 20, 4)

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
outL = Dense(3)(outL)

model = Model(inputs=[input_ss, input_sk], outputs=[outL])

model.compile(loss='mse', optimizer='adam')

date = datetime.datetime.now()
date_time = date.strftime('%m%d-%H%M')
f_path = f'./samsung/_save2/test_{date_time}' + '_{epoch:04d}_{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', patience=32, mode='auto', verbose=2, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=0, save_best_only=True, filepath=f_path)

start = time.time()
model.fit([x1_train_mm, x2_train_mm], y_train, batch_size=16, epochs=200, verbose=0, callbacks=[es, mcp], validation_split=0.25, shuffle=False)
end = time.time() - start

loss = model.evaluate([x1_test_mm, x2_test_mm], y_test)
predict = model.predict([x1_predict_mm, x2_predict_mm])
mm = model.predict([x1_realtime_mm, x2_realtime_mm])

print('----------------- minmax ----------------')
print(date.strftime('%m%d-%H%M'))
print('it took', end/60, 'minutes and', end%60,'seconds')
print('loss:',loss)
print('prediction for',y_actual[9],'is',predict[9])
print('[79000 79700 ?]:', mm)

es = EarlyStopping(monitor='val_loss', patience=32, mode='auto', verbose=2, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=0, save_best_only=True, filepath=f_path)

start = time.time()
model.fit([x1_train_ma, x2_train_ma], y_train, batch_size=16, epochs=200, verbose=0, callbacks=[es, mcp], validation_split=0.25, shuffle=False)
end = time.time() - start

loss = model.evaluate([x1_test_ma, x2_test_ma], y_test)
predict = model.predict([x1_predict_ma, x2_predict_ma])
ma = model.predict([x1_realtime_ma, x2_realtime_ma])

print('----------------- maxabs ----------------')
print(date.strftime('%m%d-%H%M'))
print('it took', end/60, 'minutes and', end%60,'seconds')
print('loss:',loss)
print('prediction for',y_actual[9],'is',predict[9])
print('[79000 79700 ?]:', ma)

es = EarlyStopping(monitor='val_loss', patience=32, mode='auto', verbose=2, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=0, save_best_only=True, filepath=f_path)

start = time.time()
model.fit([x1_train_rb, x2_train_rb], y_train, batch_size=16, epochs=200, verbose=0, callbacks=[es, mcp], validation_split=0.25, shuffle=False)
end = time.time() - start

loss = model.evaluate([x1_test_rb, x2_test_rb], y_test)
predict = model.predict([x1_predict_rb, x2_predict_rb])
rb = model.predict([x1_realtime_rb, x2_realtime_rb])

print(date.strftime('%m%d-%H%M'))
print('----------------- robust ----------------')
print('it took', end/60, 'minutes and', end%60,'seconds')
print('loss:',loss)
print('prediction for',y_actual[9],'is',predict[9])
print('[79000 79700 ?]:', rb)

