import time
import math
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score


a = np.array(range(1, 101))
x_predict = np.array(range(96,106))

def split_x(dataset, size):
    lst = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        lst.append(subset)
    return np.array(lst)    

dataset = split_x(a, 6)

x = dataset[:, :5]
x = x.reshape(x.shape[0], x.shape[1], 1)
y = dataset[:, 5]

test_set = split_x(x_predict, 6)
x_test = test_set[:, :5]
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
y_test = test_set[:, 5]

model = Sequential()
model.add(LSTM(units=32, input_shape=(5, 1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 32)                4352
_________________________________________________________________
dense (Dense)                (None, 32)                1056
_________________________________________________________________
dense_1 (Dense)              (None, 16)                528
_________________________________________________________________
dense_2 (Dense)              (None, 16)                272
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 136
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 9
=================================================================
Total params: 6,353
Trainable params: 6,353
Non-trainable params: 0
_________________________________________________________________
'''

model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', mode='min', patience=16)

start = time.time()
model.fit(x, y, epochs=300, batch_size=16, verbose=2, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)
prediction = model.predict(x_test)
r2 = r2_score(y_test, prediction)

print('it took',end,'seconds to train, accuracy(R2):',r2,', RMSE:',math.sqrt(loss))
print('prediction:',prediction,', actual data: [101 102 103 104 105 106]')

'''
it took 4.617284774780273 seconds to train, accuracy(R2): -1.0837859403283803
it stopped early with 112 epochs
-
it took 4.135490655899048 seconds to train, accuracy(R2): -2.13233851530822 , RMSE: 2.5029337834962098
prediction: [[ 99.57472 ] [100.143364] [100.656906] [101.11722 ] [101.5295  ]] , actual data: [101 102 103 104 105 106]
'''