import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

dataset = load_wine()

x = dataset.data        # (178, 13)
y = dataset.target      # (178, )
y = to_categorical(y)   # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85) # , random_state=72

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

robust_scaler = RobustScaler()
robust_scaler.fit(x_train)
x_train_rb = robust_scaler.transform(x_train)
x_test_rb = robust_scaler.transform(x_test)

power_scaler = PowerTransformer()
power_scaler.fit(x_train)
x_train_pt = power_scaler.transform(x_train)
x_test_pt = power_scaler.transform(x_test)

quantile_scaler = QuantileTransformer()
quantile_scaler.fit(x_train)
x_train_qt = quantile_scaler.transform(x_train)
x_test_qt = quantile_scaler.transform(x_test)

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=8, mode='min', verbose=2)

start = time.time()

model = Sequential()
model.add(Dense(64, input_shape=(13,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

model_std = Sequential()
model_std.add(Dense(64, input_shape=(13,), activation='relu'))
model_std.add(Dense(128, activation='relu'))
model_std.add(Dense(32, activation='relu'))
model_std.add(Dense(3, activation='softmax'))

model_std.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_std.fit(x_train_std, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

model_mm = Sequential()
model_mm.add(Dense(64, input_shape=(13,), activation='relu'))
model_mm.add(Dense(128, activation='relu'))
model_mm.add(Dense(32, activation='relu'))
model_mm.add(Dense(3, activation='softmax'))

model_mm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_mm.fit(x_train_mm, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

model_ma = Sequential()
model_ma.add(Dense(64, input_shape=(13,), activation='relu'))
model_ma.add(Dense(128, activation='relu'))
model_ma.add(Dense(32, activation='relu'))
model_ma.add(Dense(3, activation='softmax'))

model_ma.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ma.fit(x_train_ma, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

model_rb = Sequential()
model_rb.add(Dense(64, input_shape=(13,), activation='relu'))
model_rb.add(Dense(128, activation='relu'))
model_rb.add(Dense(32, activation='relu'))
model_rb.add(Dense(3, activation='softmax'))

model_rb.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_rb.fit(x_train_rb, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

model_pt = Sequential()
model_pt.add(Dense(64, input_shape=(13,), activation='relu'))
model_pt.add(Dense(128, activation='relu'))
model_pt.add(Dense(32, activation='relu'))
model_pt.add(Dense(3, activation='softmax'))

model_pt.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_pt.fit(x_train_pt, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

model_qt = Sequential()
model_qt.add(Dense(64, input_shape=(13,), activation='relu'))
model_qt.add(Dense(128, activation='relu'))
model_qt.add(Dense(32, activation='relu'))
model_qt.add(Dense(3, activation='softmax'))

model_qt.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_qt.fit(x_train_qt, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

loss = model.evaluate(x_test, y_test)
predict = model.predict(x_test)

loss_std = model_std.evaluate(x_test_std, y_test)
predict_std = model_std.predict(x_test_std)

loss_mm = model_mm.evaluate(x_test_mm, y_test)
predict_mm = model_mm.predict(x_test_mm)

loss_ma = model_ma.evaluate(x_test_ma, y_test)
predict_ma = model_ma.predict(x_test_ma)

loss_rb = model_rb.evaluate(x_test_rb, y_test)
predict_rb = model_rb.predict(x_test_rb)

loss_pt = model_pt.evaluate(x_test_pt, y_test)
predict_pt = model_pt.predict(x_test_pt)

loss_qt = model_qt.evaluate(x_test_qt, y_test)
predict_qt = model_qt.predict(x_test_qt)

end = time.time() - start

print('it took', end, 'seconds')

print('without scaling, accuracy:', loss)
print('for stdS, accuracy:', loss_std)
print('for minmax, accuracy:', loss_mm)
print('for maxabs, accuracy:', loss_ma)
print('for robust, accuracy:', loss_rb)
print('for powerT, accuracy:', loss_pt)
print('for quantileT, accuracy:', loss_qt)

'''
without scaling, accuracy: [0.5723693370819092, 0.8148148059844971]
for stdS, accuracy: [0.05777861177921295, 0.9629629850387573]
for minmax, accuracy: [0.012502740137279034, 1.0]
for maxabs, accuracy: [0.03851925581693649, 1.0]
for robust, accuracy: [0.008469696156680584, 1.0]
for powerT, accuracy: [0.015944501385092735, 1.0]
for quantileT, accuracy: [0.013760533183813095, 1.0]
-
without scaling, accuracy: [0.7559313178062439, 0.7407407164573669]
for stdS, accuracy: [0.18529173731803894, 0.9259259104728699]
for minmax, accuracy: [0.38906189799308777, 0.9259259104728699]
for maxabs, accuracy: [0.23908205330371857, 0.8888888955116272]
for robust, accuracy: [0.4997197687625885, 0.9259259104728699]
for powerT, accuracy: [0.5234611630439758, 0.9259259104728699]
for quantileT, accuracy: [0.5277317762374878, 0.8888888955116272]
'''