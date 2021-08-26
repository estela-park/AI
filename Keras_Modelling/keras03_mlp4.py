import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

x_input = numpy.array([range(10)]).transpose()
y_input = numpy.array([list(range(1, 11)), [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3], list(range(10, 0, -1))])
y_col1 = y_input[0]
y_col2 = y_input[1]
y_col3 = y_input[2]
y_input = y_input.transpose()
x_predict = numpy.array(x_input[7: 10])

model = Sequential()
# deepening the layers doesn't seem to improve the accuracy of the model
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(6))
model.add(Dense(3))

model.compile(loss='mse', optimizer='adam')
# epochs value: 1800 > 2000 > 3000 > 5000
# raising epochs to 180000 doesn't help
model.fit(x_input, y_input, epochs=1800, batch_size=1)

loss = model.evaluate(x_input, y_input)
result = model.predict(x_predict)

model_1 = Sequential()
model_1.add(Dense(4, input_dim=1))
model_1.add(Dense(5))
model_1.add(Dense(3))
model_1.add(Dense(1))

model_1.compile(loss='mse', optimizer='adam')
model_1.fit(x_input, y_col1, epochs=2000, batch_size=1)

loss = model_1.evaluate(x_input, y_col1)
y_predict_col1 = model_1.predict(x_input)

model_2 = Sequential()
model_2.add(Dense(4, input_dim=1))
model_2.add(Dense(5))
model_2.add(Dense(3))
model_2.add(Dense(1))

model_2.compile(loss='mse', optimizer='adam')
model_2.fit(x_input, y_col2, epochs=2000, batch_size=1)

loss = model_2.evaluate(x_input, y_col2)
y_predict_col2 = model_2.predict(x_input)

model_3 = Sequential()
model_3.add(Dense(4, input_dim=1))
model_3.add(Dense(5))
model_3.add(Dense(3))
model_3.add(Dense(1))

model_3.compile(loss='mse', optimizer='adam')
model_3.fit(x_input, y_col3, epochs=2000, batch_size=1)

loss = model_3.evaluate(x_input, y_col3)
y_predict_col3 = model_3.predict(x_input)

plt.scatter(x_input, y_col1, color='blue')
plt.scatter(x_input, y_col2, color='red')
plt.scatter(x_input, y_col3, color='green')
plt.plot(x_input, y_predict_col1, color='blue')
plt.plot(x_input, y_predict_col2, color='red')
plt.plot(x_input, y_predict_col3, color='green')
plt.show()

print('loss:',loss,', result for [[7], [8], [9]]:', result)

'''loss: 0.0053441692143678665 , result for [[7], [8], [9]]: [[ 8.000262   1.4387376  3.0055304]
 [ 9.001633   1.4839628  2.007371 ]
 [10.003002   1.5291882  1.0092127]]'''