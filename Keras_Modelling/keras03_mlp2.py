import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

x = [list(range(1, 11)), [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3], list(range(10, 0, -1))]
x_col1 = numpy.array(x[0]).transpose()
x_col2 = numpy.array(x[1]).transpose()
x_col3 = numpy.array(x[2]).transpose()
x_input = numpy.array(x).transpose()
y = numpy.array(list(range(11, 21)))
x_predict = numpy.array([x[0][7: 10], x[1][7: 10], x[2][7: 10]]).transpose()

model = Sequential()
model.add(Dense(4, input_dim=3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(x_input, y, epochs=1000, batch_size=1)

loss = model.evaluate(x_input, y)
result = model.predict(x_predict)

model_col1 = Sequential()
model_col1.add(Dense(4, input_dim=1))
model_col1.add(Dense(5))
model_col1.add(Dense(3))
model_col1.add(Dense(1))

model_col1.compile(loss='mse', optimizer='adam')
model_col1.fit(x_col1, y, epochs=2000, batch_size=1)

loss = model_col1.evaluate(x_col1, y)
y_predict_col1 = model_col1.predict(x_col1)

model_col2 = Sequential()
model_col2.add(Dense(4, input_dim=1))
model_col2.add(Dense(5))
model_col2.add(Dense(3))
model_col2.add(Dense(1))

model_col2.compile(loss='mse', optimizer='adam')
model_col2.fit(x_col2, y, epochs=2000, batch_size=1)

loss = model_col2.evaluate(x_col2, y)
y_predict_col2 = model_col2.predict(x_col2)

model_col3 = Sequential()
model_col3.add(Dense(4, input_dim=1))
model_col3.add(Dense(5))
model_col3.add(Dense(3))
model_col3.add(Dense(1))

model_col3.compile(loss='mse', optimizer='adam')
model_col3.fit(x_col3, y, epochs=2000, batch_size=1)

loss = model_col3.evaluate(x_col3, y)
y_predict_col3 = model_col3.predict(x_col3)

print('loss: ',loss,', result for : [[8, 9, 10], [1.5, 1.4, 1.3], [3, 2, 1]]', result)

plt.scatter(x_col1, y, color='blue')
plt.scatter(x_col2, y, color='red')
plt.scatter(x_col3, y, color='green')
plt.plot(x_col1, y_predict_col1, color='blue')
plt.plot(x_col2, y_predict_col2, color='red')
plt.plot(x_col3, y_predict_col3, color='green')
plt.show()

'''loss:  1.0247458881451621e-08 , result for [[8, 9, 10], [1.5, 1.4, 1.3], [3, 2, 1]]:  [[17.999918]
                                                                                          [18.999954]
                                                                                          [19.99999 ]]'''
