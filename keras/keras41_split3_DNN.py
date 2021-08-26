import time
import math
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score


# compared with LSTM, DNN works better with small-sized and well arranged data like those below

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
y = dataset[:, 5]

test_set = split_x(x_predict, 6)
x_test = test_set[:, :5]
y_test = test_set[:, 5]

model = Sequential()
model.add(Dense(64, input_shape=(5, )))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 64)                384
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_2 (Dense)              (None, 16)                528
_________________________________________________________________
dense_3 (Dense)              (None, 16)                272
_________________________________________________________________
dense_4 (Dense)              (None, 8)                 136
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 9
=================================================================
Total params: 3,409
Trainable params: 3,409
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
print('prediction:',prediction,', actual data: [101 102 103 104 105]')


'''
it took 4 seconds to train
accuracy(R2): 0.997218339407118 , RMSE: 0.07458767363115082
prediction: [[101.02439 ] [102.04313 ] [103.06529 ] [104.089745] [105.11421 ]]
actual data: [101 102 103 104 105]
'''