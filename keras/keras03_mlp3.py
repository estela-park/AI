import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_input = numpy.array([range(10), range(21, 31), range(201, 211)]).transpose()
y_input = numpy.array([list(range(1, 11)), [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3], list(range(10, 0, -1))]).transpose()
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

print('loss:',loss,', result for [[  7  28 208], [  8  29 209], [  9  30 210]]: ', result)

'''loss: 0.005374070256948471 , result for [[  7  28 208], [  8  29 209], [  9  30 210]]:  [[ 8.006899   1.4334244  2.9872246]
 [ 9.008673   1.4774878  1.985148 ]
 [10.010431   1.5215504  0.9830855]]'''
