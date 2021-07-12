import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from tensorflow.keras.metrics import RootMeanSquaredError as rmse

x = [list(range(1, 11)), [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3], list(range(10, 0, -1))]
x_input = numpy.array(x).transpose()
y = numpy.array(list(range(11, 21)))
x_predict = numpy.array([x[0][7: 10], x[1][7: 10], x[2][7: 10]]).transpose()

start = time.time()

model = Sequential()
model.add(Dense(4, input_dim=3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=[rmse(), 'mae'])

model.fit(x_input, y, epochs=1000, batch_size=10, verbose=1)

loss = model.evaluate(x_input, y)
result = model.predict(x_predict)

end = time.time() - start

print(end)

'''
.compile(loss='mse', optimizer='adam', metrics=['mae'])
>>>1/1 [==============================] - 0s 80ms/step - loss: 0.0015 - mae: 0.0326
.compile(loss='mse', optimizer='adam', metrics=[rmse()])
>>>1/1 [==============================] - 0s 73ms/step - loss: 1.4921e-04 - root_mean_squared_error: 0.0122
.compile(loss='mse', optimizer='adam', metrics=[rmse(), 'mae'])
>>>1/1 [==============================] - 0s 83ms/step - loss: 0.0073 - root_mean_squared_error: 0.0854 - mae: 0.0731
'''

'''
model.compile() parameter:
 - loss: displayed on screen
 - optimizer: not displayed
 - metrics: displayed and can given more than one column
'''