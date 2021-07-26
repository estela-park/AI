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

x1_predict = np.array(dataset_samsung.iloc[2561:2590, [0, 1, 2, 4]])
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

x2_predict = np.array(dataset_sk.iloc[2561:2590, [0, 1, 2, 4]])
x2_predict = split_x(x2_predict, 20)
# (14, 7, 4)

x2 = np.array(dataset_sk.iloc[:2589, [0, 1, 2, 4]])
x2 = split_x(x2, 20)
# (2579, 7, 4)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=200, random_state=72)

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

standard_scaler = StandardScaler(with_std=False)
x1_train_st = standard_scaler.fit_transform(x1_train)
x1_test_st = standard_scaler.transform(x1_test)
x1_predict_st = standard_scaler.transform(x1_predict)
x2_train_st = standard_scaler.fit_transform(x2_train)
x2_test_st = standard_scaler.transform(x2_test)
x2_predict_st = standard_scaler.transform(x2_predict)

x1_train_st = x1_train_st.reshape(2370, 20, 4)
x1_test_st = x1_test_st.reshape(200, 20, 4)
x1_predict_st = x1_predict_st.reshape(10, 20, 4)
x2_train_st = x2_train_st.reshape(2370, 20, 4)
x2_test_st = x2_test_st.reshape(200, 20, 4)
x2_predict_st = x2_predict_st.reshape(10, 20, 4)

max_abs_scaler = MaxAbsScaler()
x1_train_ma = max_abs_scaler.fit_transform(x1_train)
x1_test_ma = max_abs_scaler.transform(x1_test)
x1_predict_ma = max_abs_scaler.transform(x1_predict)
x2_train_ma = max_abs_scaler.fit_transform(x2_train)
x2_test_ma = max_abs_scaler.transform(x2_test)
x2_predict_ma = max_abs_scaler.transform(x2_predict)

x1_train_ma = x1_train_ma.reshape(2370, 20, 4)
x1_test_ma = x1_test_ma.reshape(200, 20, 4)
x1_predict_ma = x1_predict_ma.reshape(10, 20, 4)
x2_train_ma = x2_train_ma.reshape(2370, 20, 4)
x2_test_ma = x2_test_ma.reshape(200, 20, 4)
x2_predict_ma = x2_predict_ma.reshape(10, 20, 4)

robust_scaler = RobustScaler()
x1_train_rb = robust_scaler.fit_transform(x1_train)
x1_test_rb = robust_scaler.transform(x1_test)
x1_predict_rb = robust_scaler.transform(x1_predict)
x2_train_rb = robust_scaler.fit_transform(x2_train)
x2_test_rb = robust_scaler.transform(x2_test)
x2_predict_rb = robust_scaler.transform(x2_predict)

x1_train_rb = x1_train_rb.reshape(2370, 20, 4)
x1_test_rb = x1_test_rb.reshape(200, 20, 4)
x1_predict_rb = x1_predict_rb.reshape(10, 20, 4)
x2_train_rb = x2_train_rb.reshape(2370, 20, 4)
x2_test_rb = x2_test_rb.reshape(200, 20, 4)
x2_predict_rb = x2_predict_rb.reshape(10, 20, 4)

power_scaler = PowerTransformer()
x1_train_pt = power_scaler.fit_transform(x1_train)
x1_test_pt = power_scaler.transform(x1_test)
x1_predict_pt = power_scaler.transform(x1_predict)
x2_train_pt = power_scaler.fit_transform(x2_train)
x2_test_pt = power_scaler.transform(x2_test)
x2_predict_pt = power_scaler.transform(x2_predict)

x1_train_pt = x1_train_pt.reshape(2370, 20, 4)
x1_test_pt = x1_test_pt.reshape(200, 20, 4)
x1_predict_pt = x1_predict_pt.reshape(10, 20, 4)
x2_train_pt = x2_train_pt.reshape(2370, 20, 4)
x2_test_pt = x2_test_pt.reshape(200, 20, 4)
x2_predict_pt = x2_predict_pt.reshape(10, 20, 4)

quantile_scaler = QuantileTransformer()
x1_train_qt = quantile_scaler.fit_transform(x1_train)
x1_test_qt = quantile_scaler.transform(x1_test)
x1_predict_qt = quantile_scaler.transform(x1_predict)
x2_train_qt = quantile_scaler.fit_transform(x2_train)
x2_test_qt = quantile_scaler.transform(x2_test)
x2_predict_qt = quantile_scaler.transform(x2_predict)

x1_train_qt = x1_train_qt.reshape(2370, 20, 4)
x1_test_qt = x1_test_qt.reshape(200, 20, 4)
x1_predict_qt = x1_predict_qt.reshape(10, 20, 4)
x2_train_qt = x2_train_qt.reshape(2370, 20, 4)
x2_test_qt = x2_test_qt.reshape(200, 20, 4)
x2_predict_qt = x2_predict_qt.reshape(10, 20, 4)

x1_train = x1_train.reshape(2370, 20, 4)
x1_test = x1_test.reshape(200, 20, 4)
x1_predict = x1_predict.reshape(10, 20, 4)
x2_train = x2_train.reshape(2370, 20, 4)
x2_test = x2_test.reshape(200, 20, 4)
x2_predict = x2_predict.reshape(10, 20, 4)

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

model.compile(loss='mae', optimizer='adam')

import datetime
date = datetime.datetime.now()
date_time = date.strftime('%m%d-%H%M')
f_path = f'./samsung/_save/test_{date_time}' + '_{epoch:04d}_{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', patience=32, mode='auto', verbose=2, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=2, save_best_only=True, filepath=f_path)

start = time.time()
model.fit([x1_train, x2_train], y_train, batch_size=16, epochs=200, verbose=0, callbacks=[es, mcp], validation_split=0.15, shuffle=False)
end = time.time() - start

loss = model.evaluate([x1_test, x2_test], y_test)
predict = model.predict([x1_predict, x2_predict])
r2 = r2_score(y_actual, predict)

print('----------------- unscaled ----------------')
print('it took', end/60, 'minutes and', end%60,'seconds')
print('loss:',loss,'R2:',r2)
predict = model.predict([x1_predict, x2_predict])
print('prediction for',y_actual,'is',predict)

start = time.time()
model.fit([x1_train_st, x2_train_st], y_train, batch_size=16, epochs=200, verbose=2, callbacks=[es, mcp], validation_split=0.15, shuffle=False)
end = time.time() - start

loss = model.evaluate([x1_test_st, x2_test_st], y_test)
predict = model.predict([x1_predict_st, x2_predict_st])
r2 = r2_score(y_actual, predict)

print('----------------- std ----------------')
print('it took', end/60, 'minutes and', end%60,'seconds')
print('loss:',loss,'R2:',r2)
predict = model.predict([x1_predict, x2_predict])
print('prediction for',y_actual,'is',predict)

start = time.time()
model.fit([x1_train_mm, x2_train_mm], y_train, batch_size=16, epochs=200, verbose=2, callbacks=[es, mcp], validation_split=0.15, shuffle=False)
end = time.time() - start

loss = model.evaluate([x1_test_mm, x2_test_mm], y_test)
predict = model.predict([x1_predict_mm, x2_predict_mm])
r2 = r2_score(y_actual, predict)

print('----------------- minmax ----------------')
print('it took', end/60, 'minutes and', end%60,'seconds')
print('loss:',loss,'R2:',r2)
predict = model.predict([x1_predict_mm, x2_predict_mm])
print('prediction for',y_actual,'is',predict)

start = time.time()
model.fit([x1_train_ma, x2_train_ma], y_train, batch_size=16, epochs=200, verbose=2, callbacks=[es, mcp], validation_split=0.15, shuffle=False)
end = time.time() - start

loss = model.evaluate([x1_test_ma, x2_test_ma], y_test)
predict = model.predict([x1_predict_ma, x2_predict_ma])
r2 = r2_score(y_actual, predict)

print('----------------- maxabs ----------------')
print('it took', end/60, 'minutes and', end%60,'seconds')
print('loss:',loss,'R2:',r2)
predict = model.predict([x1_predict_ma, x2_predict_ma])
print('prediction for',y_actual,'is',predict)

start = time.time()
model.fit([x1_train_rb, x2_train_rb], y_train, batch_size=16, epochs=200, verbose=2, callbacks=[es, mcp], validation_split=0.15, shuffle=False)
end = time.time() - start

loss = model.evaluate([x1_test_rb, x2_test_rb], y_test)
predict = model.predict([x1_predict_rb, x2_predict_rb])
r2 = r2_score(y_actual, predict)

print('----------------- robust ----------------')
print('it took', end/60, 'minutes and', end%60,'seconds')
print('loss:',loss,'R2:',r2)
predict = model.predict([x1_predict_rb, x2_predict_rb])
print('prediction for',y_actual,'is',predict)

start = time.time()
model.fit([x1_train_qt, x2_train_qt], y_train, batch_size=16, epochs=200, verbose=2, callbacks=[es, mcp], validation_split=0.15, shuffle=False)
end = time.time() - start

loss = model.evaluate([x1_test_qt, x2_test_qt], y_test)
predict = model.predict([x1_predict_qt, x2_predict_qt])
r2 = r2_score(y_actual, predict)

print('----------------- quantileT ----------------')
print('it took', end/60, 'minutes and', end%60,'seconds')
print('loss:',loss,'R2:',r2)
predict = model.predict([x1_predict_qt, x2_predict_qt])
print('prediction for',y_actual,'is',predict)

start = time.time()
model.fit([x1_train_pt, x2_train_pt], y_train, batch_size=16, epochs=200, verbose=2, callbacks=[es, mcp], validation_split=0.15, shuffle=False)
end = time.time() - start

loss = model.evaluate([x1_test_pt, x2_test_pt], y_test)
predict = model.predict([x1_predict_pt, x2_predict_pt])
r2 = r2_score(y_actual, predict)

print('----------------- powerT ----------------')
print('it took', end/60, 'minutes and', end%60,'seconds')
print('loss:',loss,'R2:',r2)
predict = model.predict([x1_predict_pt, x2_predict_pt])
print('prediction for',y_actual,'is',predict)

'''
--------------- unscaled ----------------50
it took 6 minutes and 59 seconds
loss: 699181056.0 R2: -6641.852510499435
prediction for [80100. 80000.] is [25230.408 25251.748]
----------------- std -------------------21
it took 8 minutes and 34 seconds
loss: 796730432.0 R2: -6204.448472114704
prediction for [80100. 80000.] is [34754.832 35263.953]
----------------- minmax ----------------33
it took 4 minutes and 35 seconds
loss: 603748288.0 R2: -5395.7337260375225
prediction for [80100. 80000.] is [30748.89  31402.305]
----------------- maxabs ----------------42
it took 5 minutes and 47 seconds
loss: 743380416.0 R2: -6469.84125556041
prediction for [80100. 80000.] is [26459.693 26319.83 ]
----------------- robust ----------------38
it took 5 minutes and 17 seconds
loss: 589581952.0 R2: -5259.951856250207
prediction for [80100. 80000.] is [31456.309 31716.283]
---------------- quantileT --------------33
it took 4 minutes and 40 seconds
loss: 558334784.0 R2: -6165.073441532577
prediction for [80100. 80000.] is [27724.535 27675.027]
----------------- powerT ----------------32
it took 4 minutes and 26 seconds
loss: 352038624.0 R2: -6037.611767932073
prediction for [80100. 80000.] is [27809.564 27961.91 ]

***********************patience=50*************************
----------------- unscaled --------------70
it took 9 minutes and 55 seconds
loss: 656167616.0 R2: -6617.068598225732
prediction for [80100. 80000.] is [25627.98  25679.904]
----------------- std -------------------76
it took 10 minutes and 34 seconds
loss: 726182784.0 R2: -5539.316866056417
prediction for [80100. 80000.] is [35307.902 35639.074]
----------------- minmax ----------------51
it took 7 minutes and 10 seconds
loss: 696644800.0 R2: -6124.214406705219
prediction for [80100. 80000.] is [27719.287 27985.865]
----------------- maxabs ----------------53
it took 7 minutes and 22 seconds
loss: 732114112.0 R2: -6095.538343784878
prediction for [80100. 80000.] is [27853.56  28007.223]
----------------- robust ----------------67
it took 9 minutes and 20 seconds
loss: 602335232.0 R2: -5465.249743616721
prediction for [80100. 80000.] is [30554.021 30614.654]
----------------- quantileT -------------55
it took 7 minutes and 46 seconds
loss: 791202880.0 R2: -8129.433601785934
prediction for [80100. 80000.] is [20062.158 20094.705]


********************simpler Model***********************
----------------- std ----------------
Epoch 00085: early stopping
it took 16 minutes and 24 seconds
loss: 9785.08203125 R2: -240.31800916940222
prediction for [80800. 79900.] is [62018.27  61848.363]
----------------- minmax ----------------
Epoch 00078: early stopping
it took 15 minutes and 4 seconds
loss: 3173.079345703125 R2: -24.06981320644882
prediction for [80800. 79900.] is [75751.15  76298.836]

*******************SK-higher-dropped Model*********************
----------------- minmax ----------------
Epoch 00043: early stopping
it took 8 minutes and 22 seconds
loss: 3760.004638671875 R2: -134.16825231245312
prediction for [80800. 79900.] is [77795.625 70043.664]
----------------- maxabs ----------------
Epoch 00059: early stopping
it took 11 minutes and 6 seconds
loss: 2790.966552734375 R2: -4.473351906094502
prediction for [80800. 79900.] is [78256.84  77847.484]
----------------- robust ----------------
Epoch 00037: early stopping
it took 6 minutes and 54 seconds
loss: 2249.36474609375 R2: -5.656640484106683
prediction for [80800. 79900.] is [79062.08  77390.94 ]
----------------- quantileT ----------------
Epoch 00034: early stopping
it took 6 minutes and 17 seconds
loss: 4284.06494140625 R2: -1199.0138843653517
prediction for [80800. 79900.] is [60790.703 59249.402]
----------------- powerT ----------------
Epoch 00052: early stopping
it took 9 minutes and 29 seconds
loss: 12744.28515625 R2: -8847.34664334331
prediction for [80800. 79900.] is [25922.756 27722.47 ]
'''