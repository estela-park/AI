import time
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


x_data = np.load('../_save/_npy/k55_x_data_iris.npy')
y_data = np.load('../_save/_npy/k55_y_data_iris.npy')

y_data = to_categorical(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.85) # , random_state=72

max_abs_scaler = MaxAbsScaler()
max_abs_scaler.fit(x_train)
x_train_ma = max_abs_scaler.transform(x_train)
x_test_ma = max_abs_scaler.transform(x_test)

es = EarlyStopping(monitor='loss', patience=8, mode='min', verbose=2)

start = time.time()
model_ma = Sequential()
model_ma.add(Dense(64, input_shape=(4,), activation='relu'))
model_ma.add(Dense(128, activation='relu'))
model_ma.add(Dense(32, activation='relu'))
model_ma.add(Dense(3, activation='softmax'))

model_ma.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ma.fit(x_train_ma, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

loss_ma = model_ma.evaluate(x_test_ma, y_test)
end = time.time() - start

print('loss:',loss_ma[0],', accuracy:', loss_ma[1])

# this <simpleDNN> loss: 0.0473175607621669, accuracy: 1.0
# DNN              loss: 0.0370384678244590, accuracy: 1.0
# CNN              loss: 0.1542966216802597, accuracy: 0.9130434989929199
# LSTM             loss: 0.0912400856614112, accuracy: 0.95652174949646
# Conv1D           loss: 0.1681751906871795, accuracy: 0.9333333373069763