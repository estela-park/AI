"""
(800, 150, 150, 3) (800,)
'../_save/_npy/keras59_5_x_train'
'../_save/_npy/keras59_5_y_train'
(256, 150, 150, 3) (256,)
'../_save/_npy/keras59_5_x_test'
'../_save/_npy/keras59_5_y_test'
"""

import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

x_train = np.load('../_save/_npy/keras59_5_x_train.npy')
y_train = np.load('../_save/_npy/keras59_5_y_train.npy')
x_test = np.load('../_save/_npy/keras59_5_x_test.npy')
y_test = np.load('../_save/_npy/keras59_5_y_test.npy')


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu', input_shape=(150, 150, 3))) 
model.add(Dropout(0.5))
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu')) 
model.add(MaxPool2D())
model.add(Conv2D(16, kernel_size=(2, 2), activation='relu')) 
model.add(Dropout(0.5))
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
# time: 25.01785397529602 loss: 1.4110491275787354 ,accuracy: 0.82421875

'''
             time: 27.315396308898926 loss: 0.4783976376056671 ,accuracy: 0.7734375
0.45 -> 0.5, time: 25.01785397529602 loss: 1.4110491275787354 ,accuracy: 0.82421875

conv2d (Conv2D)              (None, 149, 149, 32)      416
_________________________________________________________________
dropout (Dropout)            (None, 149, 149, 32)      0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 148, 148, 32)      4128
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 74, 74, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 73, 73, 16)        2064
_________________________________________________________________
dropout_1 (Dropout)          (None, 73, 73, 16)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 72, 72, 16)        1040
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 36, 36, 16)        0
_________________________________________________________________
flatten (Flatten)            (None, 20736)             0
_________________________________________________________________
dense (Dense)                (None, 128)               2654336
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 65
'''