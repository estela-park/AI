import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

datasets = load_diabetes()

x = datasets.data   # (442, 10)
y = datasets.target # (442, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=72)

start = time.time()

model = Sequential()
model.add(Dense(64, input_shape=(10,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=240, validation_split=0.15, verbose=0, batch_size=32)

loss = model.evaluate(x_test, y_test)
predict = model.predict(x_test)

end = time.time() - start

r2 = r2_score(y_test, predict)

print('loss:', loss, 'time took:', end)
print('accuracy:', r2)

'''
**for random_state=72**
  10> 64> 128> 32> 1, 0.8>0.15, 240/16
  - loss: [2109.333251953125, 37.41487503051758] time took: 10.056094884872437
  - accuracy: 0.6805598196277636
10> 64> 128> 32> 1, 0.8>0.15, 240/16
accuracy: 0.6012338152021319
10> 64> 128> 32> 1, 0.8>0.15, 240/32
accuracy: 0.6172388256181822
10> 64> 128> 32> 1, 0.8>0.15, 240/32
accuracy: 0.6026845085756563
10> 80> 40> 20 >1, 0.8> 0.15, 240/32
-not good
10> 64> 128> 64> 32> 1, 0.8> 0.15, 240/32
0.5932418685038217
10> 32> 128> 32> 1, 0.8>0.15, 240/16
accuracy: 0.6044647090160546
WIthou 'relu' 10> 32> 128> 32> 1, 0.8>0.15, 240/16
accuracy: 0.6042035702457045
72 done
for random_state= 72
loss: [2156.499755859375, 37.726322174072266] time took: 8.72146201133728
accuracy: 0.6734168630230642
86 done
for random_state= 86
loss: [1759.1671142578125, 34.072715759277344] time took: 8.505428075790405
accuracy: 0.6201195170901462
'''