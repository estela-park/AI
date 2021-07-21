import numpy as np
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling1D, Conv1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping

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
model.add(Conv1D(filters=32, kernel_size=4, input_shape=(5, 1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Flatten())
model.add(Dense(1))
'''
without Flatten, dimention of layers doesn't change
_________________________________________________________________   
Layer (type)                 Output Shape              Param #      
=================================================================   
conv1d (Conv1D)              (None, 2, 32)             160
_________________________________________________________________   
dense (Dense)                (None, 2, 32)             1056
_________________________________________________________________   
dense_1 (Dense)              (None, 2, 16)             528
_________________________________________________________________   
dense_2 (Dense)              (None, 2, 16)             272
_________________________________________________________________   
dense_3 (Dense)              (None, 2, 8)              136
_________________________________________________________________   
dense_4 (Dense)              (None, 2, 1)              9
=================================================================   
with Flatten, 
_________________________________________________________________   
Layer (type)                 Output Shape              Param #      
=================================================================   
conv1d (Conv1D)              (None, 2, 32)             160
_________________________________________________________________   
dense (Dense)                (None, 2, 32)             1056
_________________________________________________________________   
dense_1 (Dense)              (None, 2, 16)             528
_________________________________________________________________   
dense_2 (Dense)              (None, 2, 16)             272
_________________________________________________________________   
dense_3 (Dense)              (None, 2, 8)              136
_________________________________________________________________   
flatten (Flatten)            (None, 16)                0
_________________________________________________________________   
dense_4 (Dense)              (None, 1)                 17
=================================================================   
'''

model.compile(loss='mse', optimizer='adam')

start = time.time()
model.fit(x, y, epochs=240, batch_size=128, verbose=2)
end = time.time() - start

loss = model.evaluate(x_test, y_test)

print('it took', end/60, 'minutes and', end%60,'seconds')
print('loss:',loss)

'''
it took 0.05636746883392334 minutes and 3.3820481300354004 seconds  
loss: 2.4351186752319336
'''