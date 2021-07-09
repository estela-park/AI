from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

x = np.array(range(100))
y = np.array(range(1, 101))


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=1)

loss = model.evaluate(x_test, y_test)
print('loss:', loss)
y_predict = model.predict([100])
print('prection for 100:', y_predict)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print('calculated accuracy(maximum_value=1):',r2)