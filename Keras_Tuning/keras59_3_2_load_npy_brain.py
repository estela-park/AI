"""
(160, 150, 150, 3) (160, )
'../_save/_npy/keras59_3_x_train'
'../_save/_npy/keras59_3_y_train': either 0 or 1
(120, 150, 150, 3) (120, )
'../_save/_npy/keras59_3_x_test'
'../_save/_npy/keras59_3_y_test': either 0 or 1
"""

import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

x_train = np.load('../_save/_npy/keras59_3_x_train.npy')
y_train = np.load('../_save/_npy/keras59_3_y_train.npy')
x_test = np.load('../_save/_npy/keras59_3_x_test.npy')
y_test = np.load('../_save/_npy/keras59_3_y_test.npy')


model = Sequential()
model.add(Conv2D(filters=4, kernel_size=(2, 2), activation='relu', input_shape=(150, 150, 3))) 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation="sigmoid"))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

start = time.time()
es = EarlyStopping(monitor='val_acc', mode='max', patience=12, verbose=2, restore_best_weights=True)
model.fit(x_train, y_train, epochs=128, validation_data=(x_test, y_test), verbose=2, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)
print('time:', end, 'loss:', loss)
# time: 4.61249852180481 loss: 0.4516060948371887, accuray: 0.8083333373069763

'''
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