from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
import time
import numpy as np
import matplotlib.pyplot as plt

# 1. data set-up
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
# (50000, 28, 28, 1) (10000, 28, 28, 1) (50000, 1) (10000, 1)

start = time.time()

x_train = x_train.reshape(60000, 28 * 28 * 1)
x_test = x_test.reshape(10000, 28 * 28 * 1)

standard_scaler = StandardScaler()
standard_scaler.fit(x_train)
x_train_std = standard_scaler.transform(x_train)
x_test_std = standard_scaler.transform(x_test)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train)
x_train_mm = min_max_scaler.transform(x_train)
x_test_mm = min_max_scaler.transform(x_test)

max_abs_scaler = MaxAbsScaler()
max_abs_scaler.fit(x_train)
x_train_ma = max_abs_scaler.transform(x_train)
x_test_ma = max_abs_scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

x_train_std = x_train_std.reshape(60000, 28, 28, 1)
x_test_std = x_test_std.reshape(10000, 28, 28, 1)

x_train_mm = x_train_mm.reshape(60000, 28, 28, 1)
x_test_mm = x_test_mm.reshape(10000, 28, 28, 1)

x_train_ma = x_train_ma.reshape(60000, 28, 28, 1)
x_test_ma = x_test_ma.reshape(10000, 28, 28, 1)

enc = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.transform(y_test).toarray() 

model = load_model('../_save/keras45_1_save_model.h5')
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2, validation_split=0.25)