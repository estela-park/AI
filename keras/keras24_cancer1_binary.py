import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

dataset = load_breast_cancer()

# print(dataset.DESCR)
# print(dataset.feature_names)

x = dataset.data   # (569, 30)
y = dataset.target # (569, )/ y.unique(): [0, 1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8) # , random_state=72

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

model_ma = Sequential()
model_ma.add(Dense(64, input_shape=(30,), activation='relu'))
model_ma.add(Dense(128, activation='relu'))
model_ma.add(Dense(32, activation='relu'))
model_ma.add(Dense(1, activation='sigmoid'))

model_ma.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ma.fit(x_train_ma, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

model_rb = Sequential()
model_rb.add(Dense(64, input_shape=(30,), activation='relu'))
model_rb.add(Dense(128, activation='relu'))
model_rb.add(Dense(32, activation='relu'))
model_rb.add(Dense(1, activation='sigmoid'))

model_rb.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_rb.fit(x_train_rb, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

model_pt = Sequential()
model_pt.add(Dense(64, input_shape=(30,), activation='relu'))
model_pt.add(Dense(128, activation='relu'))
model_pt.add(Dense(32, activation='relu'))
model_pt.add(Dense(1, activation='sigmoid'))

model_pt.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_pt.fit(x_train_pt, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

model_qt = Sequential()
model_qt.add(Dense(64, input_shape=(30,), activation='relu'))
model_qt.add(Dense(128, activation='relu'))
model_qt.add(Dense(32, activation='relu'))
model_qt.add(Dense(1, activation='sigmoid'))

model_qt.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_qt.fit(x_train_qt, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

loss_ma = model_ma.evaluate(x_test_ma, y_test)
predict_ma = model_ma.predict(x_test_ma)

loss_rb = model_rb.evaluate(x_test_rb, y_test)
predict_rb = model_rb.predict(x_test_rb)

loss_pt = model_pt.evaluate(x_test_pt, y_test)
predict_pt = model_pt.predict(x_test_pt)

loss_qt = model_qt.evaluate(x_test_qt, y_test)
predict_qt = model_qt.predict(x_test_qt)

end = time.time() - start

import matplotlib.pyplot as plt

plt.scatter(np.array(range(114)), predict_ma, color='red')
plt.scatter(np.array(range(114)), predict_rb, color='yellow')
plt.scatter(np.array(range(114)), predict_pt, color='green')
plt.scatter(np.array(range(114)), predict_qt, color='purple')
plt.scatter(np.array(range(114)), y_test, color='black')
plt.ylabel('live or die')
# plt.show()

print('for maxabs, accuracy:', loss_ma)
print('for robust, accuracy:', loss_rb)
print('for powerT, accuracy:', loss_pt)
print('for quantileT, accuracy:', loss_qt)

'''
for maxabs, accuracy: 0.9561403393745422
for robust, accuracy: 0.9649122953414917
for powerT, accuracy: 0.9561403393745422
for quantileT, accuracy: 0.9385964870452881

with Earlystopping
for maxabs, accuracy: [0.1921396255493164, 0.9649122953414917]
for robust, accuracy: [0.41252321004867554, 0.9824561476707458]
for powerT, accuracy: [0.00955619104206562, 1.0]
for quantileT, accuracy: [0.07946053147315979, 0.9649122953414917]
-
for maxabs, accuracy: [0.14990916848182678, 0.9736841917037964]
for robust, accuracy: [0.4518823027610779, 0.9473684430122375]
for powerT, accuracy: [0.24833709001541138, 0.9649122953414917]
for quantileT, accuracy: [0.09647254645824432, 0.9561403393745422]
-
for maxabs, accuracy: [0.08737146854400635, 0.9561403393745422]
for robust, accuracy: [0.26217779517173767, 0.9561403393745422]
for powerT, accuracy: [0.3079686760902405, 0.9649122953414917]
for quantileT, accuracy: [0.08165045827627182, 0.9649122953414917]
******************** prediction ********************
[0 0 0 1 1 1 1 0 0 1]
[[4.7912635e-04]
 [5.2206979e-06]
 [3.6317660e-04]
 [9.9837422e-01]
 [9.9991560e-01]
 [9.9990380e-01]
 [9.9885058e-01]
 [1.1481966e-04]
 [4.4047862e-04]
 [9.9993563e-01]]
'''