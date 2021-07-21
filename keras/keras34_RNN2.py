import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# Data setting
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])


ilayer = np.array([1, 2, 3])
print(ilayer)
x = x.reshape(4, 3, 1)

# Modeling
model = Sequential()
model.add(SimpleRNN(units=10, input_length=3, input_dim=1, activation='relu'))
#                             > timesteps     > features
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

'''
param = units(units<Hw> + features<Iw> + bias)
_________________________________________________________________   
Layer (type)                 Output Shape              Param #      
=================================================================   
simple_rnn (SimpleRNN)       (None, 10)                120
_________________________________________________________________   
dense (Dense)                (None, 10)                110
_________________________________________________________________   
dense_1 (Dense)              (None, 1)                 11
=================================================================  
'''

# Compilation & Training
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=3)

# Prediction
result = model.predict(np.array([[[5],[6],[7]]]))
print(f'prediction for [5 6 7]: {result}')

'''
epochs=100
loss: 0.1283   
prediction for [5 6 7]: [[8.62008]]
'''