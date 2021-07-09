from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
x = np.array(range(100))
y = np.array(range(1, 101))

# sequencial
x_train = x[:70]
y_train = y[:70]
x_test = x[-30:]
y_test = y[70:]
x_predict = [11]

# print(x_train.shape, y_train.shape) # (70,) (70,)
# print(x_test.shape, y_test.shape)   # (30,) (30,)

# randomized
temp = []
for i in x:
    temp.append([x[i], y[i]])

temp = np.array(temp)
np.random.shuffle(temp)
temp = temp.transpose()

x_train = temp[0][:70]
y_train = temp[1][:70]
x_test = temp[0][70]
y_test = temp[1][70]

# utilizing library
from sklearn.model_selection import train_test_split

x_train_lib, x_test_lib, y_train_lib, y_test_lib = train_test_split(x, y, train_size=0.7, shuffle=True)

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1)

loss = model.evaluate(x_test, y_test)
print('loss:', loss)
result = model.predict(x_predict)
print('result for 11:', result)
y_predict = model.predict(x_predict)
