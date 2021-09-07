import time
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


x_data = np.load('../_save/_npy/k55_x_data_cancer.npy')
y_data = np.load('../_save/_npy/k55_y_data_cancer.npy')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8)

max_abs_scaler = MaxAbsScaler()
max_abs_scaler.fit(x_train)
x_train_ma = max_abs_scaler.transform(x_train)
x_test_ma = max_abs_scaler.transform(x_test)

start = time.time()
es = EarlyStopping(monitor='loss', patience=8, mode='min', verbose=2)
model_ma = Sequential()
model_ma.add(Dense(64, input_shape=(30,), activation='relu'))
model_ma.add(Dense(128, activation='relu'))
model_ma.add(Dense(32, activation='relu'))
model_ma.add(Dense(1, activation='sigmoid'))

model_ma.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ma.fit(x_train_ma, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])
loss_ma = model_ma.evaluate(x_test_ma, y_test)
end = time.time() - start

print('this <Conv1D> took',end,'loss:',loss_ma[0],'accuracy:',loss_ma[1])

# this <Conv1D> took 5 secs loss: 0.0747335776686668, accuracy: 0.9561403393745422
# DNN                       loss: 0.4125232100486755, accuracy: 0.9824561476707458
# CNN                       loss: 0.3068947792053222, accuracy: 0.930232584476471
# LSTM                      loss: 0.9253537654876709, accuracy: 0.8720930218696594
# Conv1D                    loss: 0.2286279201507568, accuracy: 0.9186046719551086