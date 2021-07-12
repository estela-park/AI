from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time
from sklearn.model_selection import train_test_split


x = np.array(range(1,16))
y = np.array(range(1,16))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8)

start = time.time()

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=10, verbose=2, validation_split=0.15)

loss = model.evaluate(x_test, y_test)

end = time.time() - start

y_predict = model.predict(x_test)
print('actual data:', y_test)
print('predicted:', y_predict)