import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x_train = np.array(range(1, 8))
y_train = np.array(range(1, 8))
x_test = np.array(range(8, 11))
# y_test = np.array(range(8, 11))
x_validation = np.array(range(11,14))
y_validation = np.array(range(11,14))

start = time.time()

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=2, validation_data=(x_validation, y_validation))

# loss = model.evaluate(x_test, y_test)

end = time.time() - start

y_predict = model.predict(x_test)
# print('actual data:', y_test)
print('predicted:', y_predict)

