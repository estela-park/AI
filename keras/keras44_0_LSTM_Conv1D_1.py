# Training time-series data with Conv1D instead of LSTM
# and You can do that!

import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D
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
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

'''
*** without Flatten, dimention of layers doesn't change ***
adding Dense(1) layer will only shorten the number of nodes to be (None, ... , 1)
'''

model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', mode='min', patience=8, verbose=2, restore_best_weights=True)

start = time.time()
model.fit(x, y, epochs=240, batch_size=128, verbose=2, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)

print('it took', end/60, 'minutes and', end%60,'seconds')
print('loss:',loss)


'''

'''