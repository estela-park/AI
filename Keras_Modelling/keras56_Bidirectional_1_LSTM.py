import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional

# Data setting
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])


ilayer = np.array([1, 2, 3])
print(ilayer)
x = x.reshape(4, 3, 1)

# Modeling
model = Sequential()
model.add(LSTM(units=10, input_shape=(3, 1), return_sequences=True))
model.add(Bidirectional(LSTM(units=10, input_shape=(3, 1), return_sequences=True)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

# Compilation & Training
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=3)

# Prediction
result = model.predict(np.array([[[5],[6],[7]]]))
print(f'prediction for [5 6 7]: {result}')

'''
it works
    model.add(LSTM(return_sequences=True))
    model.add(Bidirectional(LSTM(return_sequences=True)))
it doesn't work
    model.add(Bidirectional(LSTM(return_sequences=True)))
    model.add(LSTM(return_sequences=True))
Resons
    input_shape
'''