import time
from sklearn.datasets import load_iris
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

dataset = load_iris()

x = dataset.data   # (150, 4)
y = dataset.target # (150, )
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85) # , random_state=72

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
model_ma.add(Dense(64, input_shape=(4,), activation='relu'))
model_ma.add(Dense(128, activation='relu'))
model_ma.add(Dense(32, activation='relu'))
model_ma.add(Dense(3, activation='softmax'))

model_ma.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ma.fit(x_train_ma, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

model_rb = Sequential()
model_rb.add(Dense(64, input_shape=(4,), activation='relu'))
model_rb.add(Dense(128, activation='relu'))
model_rb.add(Dense(32, activation='relu'))
model_rb.add(Dense(3, activation='softmax'))

model_rb.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_rb.fit(x_train_rb, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

model_pt = Sequential()
model_pt.add(Dense(64, input_shape=(4,), activation='relu'))
model_pt.add(Dense(128, activation='relu'))
model_pt.add(Dense(32, activation='relu'))
model_pt.add(Dense(3, activation='softmax'))

model_pt.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_pt.fit(x_train_pt, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

model_qt = Sequential()
model_qt.add(Dense(64, input_shape=(4,), activation='relu'))
model_qt.add(Dense(128, activation='relu'))
model_qt.add(Dense(32, activation='relu'))
model_qt.add(Dense(3, activation='softmax'))

model_qt.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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

print('for maxabs, accuracy:', loss_ma)
print('for robust, accuracy:', loss_rb)
print('for powerT, accuracy:', loss_pt)
print('for quantileT, accuracy:', loss_qt)

'''
for maxabs, accuracy: [0.037038467824459076, 1.0]
for robust, accuracy: [0.2522203326225281, 0.95652174949646]
for powerT, accuracy: [0.2688288986682892, 0.95652174949646]
for quantileT, accuracy: [0.10831735283136368, 0.9130434989929199]
-
for maxabs, accuracy: [0.04239023104310036, 1.0]
for robust, accuracy: [0.08042015880346298, 0.95652174949646]
for powerT, accuracy: [0.19658298790454865, 0.95652174949646]
for quantileT, accuracy: [0.051589157432317734, 0.95652174949646]
-
for maxabs, accuracy: [0.039518505334854126, 1.0]
for robust, accuracy: [0.03774683177471161, 0.95652174949646]
for powerT, accuracy: [0.031904108822345734, 1.0]
for quantileT, accuracy: [0.04313637316226959, 1.0]

**how predict looks like:
    -[[9.9904639e-01 9.5356849e-04 3.0061298e-09]
      [9.9844283e-01 1.5571439e-03 1.1436463e-08]
      [1.1278500e-03 9.9623919e-01 2.6329122e-03]
    -max element be 1, others 0 -> one-hot vector

******************** prediction ********************
[[0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]]

[[4.5567073e-04 9.9869651e-01 8.4788178e-04]
 [1.3566726e-04 5.6011450e-01 4.3974984e-01]
 [9.9964952e-01 3.5048270e-04 5.5298661e-11]
 [4.2513723e-04 9.9808609e-01 1.4887643e-03]
 [9.9953866e-01 4.6134347e-04 1.2989168e-10]
 [6.4218487e-04 9.9848735e-01 8.7043823e-04]
 [1.2246167e-05 1.2929359e-01 8.7069416e-01]
 [9.9908304e-01 9.1696315e-04 2.6099031e-10]
 [2.1305690e-03 9.9744070e-01 4.2874293e-04]
 [3.7074875e-04 9.7011495e-01 2.9514337e-02]]
'''