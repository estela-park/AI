from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score


x = [1, 2, 3, 4, 5]
y = [1, 2, 4, 3, 5]
x_pred = [6]

x_input = np.array(x)
y_input = np.array(y)

model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(8))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_input, y_input, epochs=1350, batch_size=1)

loss = model.evaluate(x_input, y_input)
print('loss: ', loss)
result = model.predict(x_input)
r2 = r2_score(y_input, result)
print('result: ', result)
print('accuracy(in R-squared value):', r2)

'''result:  [[1.2176285]
 [2.1109326]
 [3.0042367]
 [3.8975408]
 [4.7908444]]
accuracy(in R-squared value): 0.8099461702991789'''

# Assignment
# Raise the R2 score