import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

x = [list(range(1, 11)), [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3], list(range(10, 0, -1))]
x_input = numpy.array(x).transpose()
y = numpy.array(list(range(11, 21)))
x_predict = numpy.array([x[0][7: 10], x[1][7: 10], x[2][7: 10]]).transpose()

model = Sequential()
model.add(Dense(4, input_dim=3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

start = time.time()
model.compile(loss='mse', optimizer='adam')

model.fit(x_input, y, epochs=1000, batch_size=10, verbose=1)


loss = model.evaluate(x_input, y)
result = model.predict(x_predict)

end = time.time() - start

print(end)

'''
verbose=0: 2.6704587936401367
verbose=1: 3.901326894760132
verbose=2: 3.3681464195251465
verbose=3: 3.3207240104675293
'''

'''
controlled: verbose=1, 1<=batch_size<=10
batch_size=1: 19.114025354385376
batch_size=2: 11.768780708312988
batch_size=5: 5.6235432624816895
batch_size=10: 3.8073935508728027
'''