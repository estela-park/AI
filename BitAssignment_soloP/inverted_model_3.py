import datetime
import numpy as np
import pandas as pd
import time
from tensorflow.keras.layers import LSTM, Conv1D, Dropout, Dense, Input, concatenate, MaxPool1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

def split_x(dataset, size):
    lst = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        lst.append(subset)
    return np.array(lst)   


# PusanCases: 4/1 ~ 8/3, x1: previous 10 days, label: cases count for the 11th day
data_PC = pd.read_csv('../_save/_solo/cases_pusan.csv', encoding='EUC-KR')
# menually splitting train - validation data
x1 = np.array(data_PC['0'])
x1 = split_x(x1, 10)[20:-1]
x1 = x1.reshape(95, 10, 1)
# 4/21 ~ 7/24 (95, 10, 1)
y = np.array(data_PC['0'])[30:]
# 5/1 ~ 8/3 (95,)

# SeoulCases: 4/17 ~ 8/3
# Traffic from Seoul to Pusan: 4/1 ~ 7/31 <4/2 missing>
# x2: dot product of SeoulCases and Traffic < 1, 3, 5, 7 days before>
data_SC = pd.read_csv('../_save/_solo/cases_seoul.csv', encoding='EUC-KR')
data_TF = pd.read_csv('../_save/_solo/traffic.csv', encoding='EUC-KR')
data_SC = np.array(data_SC['0'])
# 4/17 ~ 8/3
data_TF = np.array(data_TF['도착지방향총교통량'])
# 4/1 ~ 7/31, 4/2 missing

# multiplying with weight
data_2 = data_SC[:-3] * data_TF[15:]
# 4/17 ~ 7/31 (106,)
x2 = np.array(data_2[10:-1]).reshape(95, 1)
# 4/28 ~ 7/31 (95,)

# PusanVaccination: 4/17 ~ 8/3, x3
data_PV = pd.read_csv('../_save/_solo/vax.csv', encoding='EUC-KR')
x3 = np.array(data_PV['0'])[:-14].reshape(95, 1)
# 4/17 ~ 7/20 (95,)


# Modelling
input_x1 = Input(shape=(10, 1))
hl_x1 = Conv1D(48, 2, activation='relu', padding='same')(input_x1)
hl_x1 = MaxPool1D()(hl_x1)
hl_x1 = Dropout(0.2)(hl_x1)
hl_x1 = LSTM(24,activation='relu')(hl_x1)

input_x2 = Input(shape=(1))
hl_x2 = Dense(24, activation='relu')(input_x2)
hl_x2 = Dropout(0.35)(hl_x2)

input_x3 = Input(shape=(1))
hl_x3 = Dense(24, activation='relu')(input_x3)
hl_x3 = Dropout(0.35)(hl_x3)

outL = concatenate([hl_x1, hl_x2, hl_x3])
outL = Dense(24, activation='relu')(outL)
outL = Dense(6, activation='relu')(outL)
outL = Dense(1)(outL)

model = Model(inputs=[input_x1, input_x2, input_x3], outputs=[outL])

# Compilation & Fitting
# loss: mse, optimizer: adam, batch: 16, validation: 0.15
model.compile(loss='mse', optimizer='adam')

for i in range(200):
    date = datetime.datetime.now()
    date_time = date.strftime('%m%d-%H%M')
    f_path = f'../_save/_solo/test_{date_time}_{i}.h5'

    es = EarlyStopping(monitor='val_loss', patience=36, mode='auto', verbose=2, restore_best_weights=True)

    start = time.time()
    model.fit([x1, x2, x3], y, batch_size=4, epochs=360, verbose=0, callbacks=[es], validation_split=0.15)
    end = time.time() - start

    print('it took', end/60, 'minutes and', end%60,'seconds')
    print('now is',date.strftime('%m%d-%H%M'))

    loss = model.evaluate([x1, x2, x3], y)
    if loss < 30000:
        model.save(f_path)

    print('loss:',loss)