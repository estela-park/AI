import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import r2_score

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 3, 5])

input1 = Input(shape=(1,))
dense1 = Dense(16, activation='relu')(input1)
dense2 = Dense(8)(dense1)
dense3 = Dense(4)(dense2)
dense4 = Dense(2)(dense3)
output1 = Dense(1)(dense4)

model = Model(inputs=input1, outputs=output1)

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1200, batch_size=2, verbose=2)

predict = model.predict(x)
print(y)
print(predict)
r2 = r2_score(y, predict)
print('r2:', r2)

'''
r2: 0.8098820299364377
'''