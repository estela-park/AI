import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

datasets = load_diabetes()

x = datasets.data   # (442, 10)
y = datasets.target # (442, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85)

start = time.time()

model = Sequential()
model.add(Dense(5, input_shape=(10,)))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=1800, validation_split=0.15, verbose=2, batch_size=32)

loss = model.evaluate(x_test, y_test)
predict = model.predict(x_test)

end = time.time() - start

print('loss:', loss, 'actual data:', y_test, 'machine predicted:', predict)