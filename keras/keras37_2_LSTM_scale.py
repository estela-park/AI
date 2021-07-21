import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.core import Dropout

# Data setting
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_test = np.array([[[50],[60],[70]]])

x = x.reshape(13, 3, 1)

# Modeling
model = Sequential()
model.add(LSTM(units=32, input_shape=(3, 1), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
# model.add(Dense(15, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()

'''
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 10)                480
_________________________________________________________________
dense (Dense)                (None, 10)                110
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11
'''

# Compilation & Training
start = time.time()
es = EarlyStopping(monitor='val_loss', mode='min', patience=24, verbose=2)
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=3, validation_split = 0.15, callbacks=[es])
end = time.time() - start

# Prediction
result = model.predict(x_test)
print(f'prediction for [50 60 70]: {result}')

'''
unit=5
RNN1-DNN7
prediction for [50 60 70]: [[75.6667]]
RNN1-DNN6
prediction for [50 60 70]: [[74.211494]]
unit=30
prediction for [50 60 70]: [[74.796616]]
unit=32
prediction for [50 60 70]: [[74.90064]]
with 'relu' and ES
prediction for [50 60 70]: [[112.16554]] *stopped at 45
adding Dropout and taming DNN layers
prediction for [50 60 70]: [[83.86038]] *stopped at 39
'''