from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

x_input = np.array(range(1, 11))
y_input = np.array([1, 2, 4, 3, 5, 7, 9, 3, 8, 12])
x_predict = [6]

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_input, y_input, epochs=500, batch_size=1)

loss = model.evaluate(x_input, y_input)
print('loss:', loss)
result = model.predict(x_predict)
print('result for 11:', result)
y_predict = model.predict(x_input)
print('prediction for x_input:', y_predict)

plt.scatter(x_input, y_input)
plt.plot(x_input, y_predict, color='blue')
plt.show()