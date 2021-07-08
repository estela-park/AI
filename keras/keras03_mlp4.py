import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_input = numpy.array([range(10)]).transpose()
print(x_input)
y_input = numpy.array([list(range(1, 11)), [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3], list(range(10, 0, -1))]).transpose()
print(y_input)
x_predict = numpy.array(x_input[7: 10])
print(x_predict)

model = Sequential()
# deepening the layers doesn't seem to improve the accuracy of the model
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(6))
model.add(Dense(3))

model.compile(loss='mse', optimizer='adam')
# epochs value: 1800 > 2000 > 3000 > 5000
# raising epochs to 180000 doesn't help
model.fit(x_input, y_input, epochs=180000, batch_size=1)

loss = model.evaluate(x_input, y_input)
result = model.predict(x_predict)

print('loss:',loss,', result for [[7], [8], [9]]:', result)

'''loss: 0.0053441692143678665 , result for [[7], [8], [9]]: [[ 8.000262   1.4387376  3.0055304]
 [ 9.001633   1.4839628  2.007371 ]
 [10.003002   1.5291882  1.0092127]]'''