import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

x = [list(range(1, 11)), [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3], list(range(10, 0, -1))]
x_input = numpy.array(x).transpose()
y = numpy.array(list(range(11, 21)))
x_predict = numpy.array([x[0][7: 10], x[1][7: 10], x[2][7: 10]]).transpose()

# when input dimension is vector, arg goes input_shape=()
model = Sequential()
model.add(Dense(4, input_shape=(3, )))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(x_input, y, epochs=1000, batch_size=1)

loss = model.evaluate(x_input, y)
result = model.predict(x_predict)
print('loss:',loss,'result:',result)
