from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

x = np.array(range(1, 11))
y = np.array(range(1, 11))
x_train = x[0:7]
y_train = y[0:7]
x_test = x[7:]
y_test = y[7:]
x_predict = [11]

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1)

loss = model.evaluate(x_test, y_test)
print('loss:', loss)
result = model.predict(x_predict)
print('result for 11:', result)
y_predict = model.predict(x_predict)
