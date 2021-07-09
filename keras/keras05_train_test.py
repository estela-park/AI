from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array(range(1, 8))
y_train = np.array(range(1, 8))
x_test = np.array(range(8, 11))
y_test = np.array(range(8, 11))
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
# print('prediction for x_input:', y_predict)

# plt.scatter(x_input, y_input)
# plt.plot(x_input, y_predict, color='blue')
# plt.show()
'''loss: 1.515824466814808e-12
result for 11: [[11.]]'''