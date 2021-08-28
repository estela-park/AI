"""
(8005, 150, 150, 3) (8005,)
'../_save/_npy/keras59_4_x_train'
'../_save/_npy/keras59_4_y_train'
(2023, 150, 150, 3) (2023,)
'../_save/_npy/keras59_4_x_test'
'../_save/_npy/keras59_4_y_test'
"""

import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

x_train = np.load('../_save/_npy/keras59_4_x_train.npy')
y_train = np.load('../_save/_npy/keras59_4_y_train.npy')
x_test = np.load('../_save/_npy/keras59_4_x_test.npy')
y_test = np.load('../_save/_npy/keras59_4_y_test.npy')


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu', input_shape=(150, 150, 3))) 
model.add(Dropout(0.45))
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu')) 
model.add(MaxPool2D())
model.add(Conv2D(16, kernel_size=(2, 2), activation='relu')) 
model.add(Dropout(0.45))
model.add(Conv2D(16, kernel_size=(2, 2), activation='relu')) 
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation="sigmoid"))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

start = time.time()
es = EarlyStopping(monitor='val_acc', mode='max', patience=12, verbose=2, restore_best_weights=True)
model.fit(x_train, y_train, epochs=128, validation_data=(x_test, y_test), verbose=2, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)
print('time:', end, 'loss:', loss[0], ',accuracy:', loss[1])
# time: 112.34202337265015 loss: 0.6165000200271606 ,accuracy: 0.6589223742485046

'''
frozen                       time: 44.78793525695801 loss: 0.6931477189064026, accuracy: 0.5002471804618835
add Conv layer,              still not thawed
add 2 Conv with MP Drop out, worked

conv2d (Conv2D)              (None, 149, 149, 4)       52
_________________________________________________________________
flatten (Flatten)            (None, 88804)             0
_________________________________________________________________
dense (Dense)                (None, 128)               11367040
_________________________________________________________________
dense_1 (Dense)              (None, 32)                4128
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 264
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 9
'''