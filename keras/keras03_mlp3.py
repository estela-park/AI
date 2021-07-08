import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

x_input = numpy.array([range(10), range(21, 31), range(201, 211)])
x_col1 = x_input[0]
x_col2 = x_input[1]
x_col3 = x_input[2]
x_input = x_input.transpose()
y_input = numpy.array([list(range(1, 11)), [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3], list(range(10, 0, -1))])
y_col1 = y_input[0]
y_col2 = y_input[1]
y_col3 = y_input[2]
y_input = y_input.transpose()
x_predict = numpy.array(x_input[7: 10])


model = Sequential()
# deepening the layers doesn't seem to improve the accuracy of the model
model.add(Dense(5, input_dim=3))
model.add(Dense(3))
model.add(Dense(6))
model.add(Dense(3))

model.compile(loss='mse', optimizer='adam')
# epochs value: 2000 > 3000 > 5000
model.fit(x_input, y_input, epochs=1800, batch_size=1)

loss = model.evaluate(x_input, y_input)
result = model.predict(x_predict)

model_col1_1 = Sequential()
model_col1_1.add(Dense(4, input_dim=1))
model_col1_1.add(Dense(5))
model_col1_1.add(Dense(3))
model_col1_1.add(Dense(1))

model_col1_1.compile(loss='mse', optimizer='adam')
model_col1_1.fit(x_col1, y_col1, epochs=2000, batch_size=1)

loss = model_col1_1.evaluate(x_col1, y_col1)
y_predict_col1_1 = model_col1_1.predict(x_col1)

model_col1_2 = Sequential()
model_col1_2.add(Dense(4, input_dim=1))
model_col1_2.add(Dense(5))
model_col1_2.add(Dense(3))
model_col1_2.add(Dense(1))

model_col1_2.compile(loss='mse', optimizer='adam')
model_col1_2.fit(x_col1, y_col2, epochs=2000, batch_size=1)

loss = model_col1_2.evaluate(x_col1, y_col2)
y_predict_col2_1 = model_col1_2.predict(x_col1)

model_col1_3 = Sequential()
model_col1_3.add(Dense(4, input_dim=1))
model_col1_3.add(Dense(5))
model_col1_3.add(Dense(3))
model_col1_3.add(Dense(1))

model_col1_3.compile(loss='mse', optimizer='adam')
model_col1_3.fit(x_col1, y_col3, epochs=2000, batch_size=1)

loss = model_col1_3.evaluate(x_col1, y_col3)
y_predict_col3_1 = model_col1_3.predict(x_col1)

plt.scatter(x_col1, y_col1, color='blue')
plt.scatter(x_col1, y_col2, color='red')
plt.scatter(x_col1, y_col3, color='green')
plt.plot(x_col1, y_predict_col1_1, color='blue')
plt.plot(x_col1, y_predict_col2_1, color='red')
plt.plot(x_col1, y_predict_col3_1, color='green')
plt.show()

model_col2_1 = Sequential()
model_col2_1.add(Dense(4, input_dim=1))
model_col2_1.add(Dense(5))
model_col2_1.add(Dense(3))
model_col2_1.add(Dense(1))

model_col2_1.compile(loss='mse', optimizer='adam')
model_col2_1.fit(x_col2, y_col1, epochs=2000, batch_size=1)

loss = model_col2_1.evaluate(x_col2, y_col1)
y_predict_col1_2 = model_col2_1.predict(x_col2)

model_col2_2 = Sequential()
model_col2_2.add(Dense(4, input_dim=1))
model_col2_2.add(Dense(5))
model_col2_2.add(Dense(3))
model_col2_2.add(Dense(1))

model_col2_2.compile(loss='mse', optimizer='adam')
model_col2_2.fit(x_col2, y_col2, epochs=2000, batch_size=1)

loss = model_col2_2.evaluate(x_col2, y_col2)
y_predict_col2_2 = model_col2_2.predict(x_col2)

model_col2_3 = Sequential()
model_col2_3.add(Dense(4, input_dim=1))
model_col2_3.add(Dense(5))
model_col2_3.add(Dense(3))
model_col2_3.add(Dense(1))

model_col2_3.compile(loss='mse', optimizer='adam')
model_col2_3.fit(x_col2, y_col3, epochs=2000, batch_size=1)

loss = model_col2_3.evaluate(x_col2, y_col3)
y_predict_col3_2 = model_col2_3.predict(x_col2)

plt.scatter(x_col2, y_col1, color='blue')
plt.scatter(x_col2, y_col2, color='red')
plt.scatter(x_col2, y_col3, color='green')
plt.plot(x_col2, y_predict_col1_2, color='blue')
plt.plot(x_col2, y_predict_col2_2, color='red')
plt.plot(x_col2, y_predict_col3_2, color='green')
plt.show()

model_col3_1 = Sequential()
model_col3_1.add(Dense(4, input_dim=1))
model_col3_1.add(Dense(5))
model_col3_1.add(Dense(3))
model_col3_1.add(Dense(1))

model_col3_1.compile(loss='mse', optimizer='adam')
model_col3_1.fit(x_col3, y_col1, epochs=2000, batch_size=1)

loss = model_col3_1.evaluate(x_col3, y_col1)
y_predict_col1_3 = model_col3_1.predict(x_col3)

model_col3_2 = Sequential()
model_col3_2.add(Dense(4, input_dim=1))
model_col3_2.add(Dense(5))
model_col3_2.add(Dense(3))
model_col3_2.add(Dense(1))

model_col3_2.compile(loss='mse', optimizer='adam')
model_col3_2.fit(x_col3, y_col2, epochs=2000, batch_size=1)

loss = model_col3_2.evaluate(x_col3, y_col2)
y_predict_col2_3 = model_col3_2.predict(x_col3)

model_col3_3 = Sequential()
model_col3_3.add(Dense(4, input_dim=1))
model_col3_3.add(Dense(5))
model_col3_3.add(Dense(3))
model_col3_3.add(Dense(1))

model_col3_3.compile(loss='mse', optimizer='adam')
model_col3_3.fit(x_col3, y_col3, epochs=2000, batch_size=1)

loss = model_col3_3.evaluate(x_col3, y_col3)
y_predict_col3_3 = model_col3_3.predict(x_col3)

plt.scatter(x_col3, y_col1, color='blue')
plt.scatter(x_col3, y_col2, color='red')
plt.scatter(x_col3, y_col3, color='green')
plt.plot(x_col3, y_predict_col1_3, color='blue')
plt.plot(x_col3, y_predict_col2_3, color='red')
plt.plot(x_col3, y_predict_col3_3, color='green')
plt.show()

print('loss:',loss,', result for [[  7  28 208], [  8  29 209], [  9  30 210]]: ', result)

'''loss: 0.005374070256948471 , result for [[  7  28 208], [  8  29 209], [  9  30 210]]:  [[ 8.006899   1.4334244  2.9872246]
 [ 9.008673   1.4774878  1.985148 ]
 [10.010431   1.5215504  0.9830855]]'''
