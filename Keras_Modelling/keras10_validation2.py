from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time


x = np.array(range(1,14))
y = np.array(range(1,14))
x_train = x[:7]
y_train = y[:7]
x_test = x[7:10]
y_test = y[7:10]
x_validation = x[10:]
y_validation = y[10:]

start = time.time()

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=2, validation_data=(x_validation, y_validation))

loss = model.evaluate(x_test, y_test)

end = time.time() - start

y_predict = model.predict(x_test)
print('actual data:', y_test)
print('predicted:', y_predict)