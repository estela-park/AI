import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Data setting
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])


ilayer = np.array([1, 2, 3])
print(ilayer)
x = x.reshape(4, 3, 1)

# Modeling
model = Sequential()
model.add(LSTM(units=10, input_shape=(3, 1)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

'''
param = (units(units<Hw> + features<Iw> + bias)) * 4
                                                   > forget_gate, input_gate, relevance_gate, output_gate 
_________________________________________________________________ 
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 10)                520
_________________________________________________________________
dense (Dense)                (None, 10)                110
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11
'''

# Compilation & Training
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=3)

# Prediction
result = model.predict(np.array([[[5],[6],[7]]]))
print(f'prediction for [5 6 7]: {result}')