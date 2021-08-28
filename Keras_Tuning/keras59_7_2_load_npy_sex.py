"""
(2800, 150, 150, 3) (2800, )
'../_save/_npy/keras59_7_x_train'
'../_save/_npy/keras59_7_y_train'
(509, 150, 150, 3) (509, )
'../_save/_npy/keras59_7_x_test'
'../_save/_npy/keras59_7_y_test'
"""

import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

x_train = np.load('../_save/_npy/keras59_7_x_train.npy')
y_train = np.load('../_save/_npy/keras59_7_y_train.npy')
x_test = np.load('../_save/_npy/keras59_7_x_test.npy')
y_test = np.load('../_save/_npy/keras59_7_y_test.npy')


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
es = EarlyStopping(monitor='val_loss', mode='min', patience=12, verbose=2, restore_best_weights=True)
model.fit(x_train, y_train, epochs=128, validation_data=(x_test, y_test), verbose=2, callbacks=[es])
end = time.time() - start

# 0: men, 1: women
loss = model.evaluate(x_test, y_test)
print('time:', end, 'loss:', loss)
# time: 16.058151483535767 loss: 0.682204008102417, accuracy: 0.5913556218147278

'''
time: 14.646761894226074 loss: [1.226414680480957, 0.4950883984565735]

conv2d (Conv2D)              (None, 149, 149, 8)       104
_________________________________________________________________
dropout (Dropout)            (None, 149, 149, 8)       0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 148, 148, 4)       132
_________________________________________________________________
dropout_1 (Dropout)          (None, 148, 148, 4)       0
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 74, 74, 4)         0
_________________________________________________________________
flatten (Flatten)            (None, 21904)             0
_________________________________________________________________
dense (Dense)                (None, 512)               11215360
_________________________________________________________________
dense_1 (Dense)              (None, 128)               65664
_________________________________________________________________
dense_2 (Dense)              (None, 32)                4128
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 264
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 9
'''

'''
with activation='selu', time: 13.084634065628052 loss: [0.7822234034538269, 0.5638507008552551]
with activation='relu', time: 11.737024784088135 loss: [0.6625196933746338, 0.6365422606468201]
    > batch_size=16,    time: 17.655009746551514 loss: [0.6718940734863281, 0.5717092156410217]
                        time: 21.758676290512085 loss: [0.6776601076126099, 0.5972495079040527]
    > batch_size=64,    time: 13.589915037155151 loss: [0.6905028820037842, 0.5363457798957825] - froze
                        time: 12.218546628952026 loss: [0.683356523513794, 0.5461689829826355]  - thawed
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

'''
time: 14.74093770980835 loss: [0.6905161142349243, 0.5363457798957825]

conv2d (Conv2D)              (None, 149, 149, 16)      208
_________________________________________________________________
flatten (Flatten)            (None, 355216)            0
_________________________________________________________________
dense (Dense)                (None, 32)                11366944
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 264
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 9
'''