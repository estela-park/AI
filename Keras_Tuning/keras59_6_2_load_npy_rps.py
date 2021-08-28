"""
(2000, 150, 150, 3) (2000, 3)_one-hot encoded
'../_save/_npy/keras59_6_x_train'
'../_save/_npy/keras59_6_y_train'
(520, 150, 150, 3) (520, 3)_one-hot encoded
'../_save/_npy/keras59_6_x_test'
'../_save/_npy/keras59_6_y_test'
"""
# loading data.npy, data is not fed by generator. 
# steps_per_epoch doesn't do anything, it used when model.fit(data=DirectoryIterator)


import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

x_train = np.load('../_save/_npy/keras59_6_x_train.npy')
y_train = np.load('../_save/_npy/keras59_6_y_train.npy')
x_test = np.load('../_save/_npy/keras59_6_x_test.npy')
y_test = np.load('../_save/_npy/keras59_6_y_test.npy')

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
model.add(Dense(3, activation="softmax"))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

start = time.time()
es = EarlyStopping(monitor='val_acc', mode='max', patience=12, verbose=2, restore_best_weights=True)
model.fit(x_train, y_train, epochs=128, validation_data=(x_test, y_test), verbose=2, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)
print('time:', end, 'loss:', loss)
# time: 40.417396783828735 loss: 1.487898588180542, accuracy: 0.5711538195610046

'''
** FC's complexcity doesn't seem to help acc improve

es monitors val_acc,   time: 47.58834385871887 loss: 3.648242473602295, accuracy: 0.5519230961799622
one less layer for FC, time: 33.02543878555298 loss: 3.601475954055786, accuracy: 0.5326923131942749
addding 2 Dropout,     time: 30.972466707229614 loss: 1.5916645526885986, accuracy: 0.5096153616905212
Drop 0.15 -> 0.25      time: 43.471883058547974 loss: 2.163599729537964, accuracy: 0.5192307829856873
Drop 0.25 -> 0.35      time: 48.59169912338257 loss: 2.5914742946624756, accuracy: 0.5307692289352417
Drop 0.35 -> 0.45      time: 39.18045926094055 loss: 1.1649584770202637, accuracy: 0.5519230961799622
less filters, 0.15,    time: 55.789459466934204 loss: 3.508694887161255, accuracy: 0.4961538314819336
Drop 0.15 -> 0.35      time: 27.49142360687256 loss: 1.0539636611938477, accuracy: 0.4307692348957062
Drop 0.35 -> 0.45      time: 36.58832335472107 loss: 1.6356122493743896, accuracy: 0.48461538553237915
more filters, 0.25,    time: 60.13966131210327 loss: 3.3451850414276123, accuracy: 0.4923076927661896
Drop 0.25 -> 0.45      time: 41.0141716003418 loss: 1.7910631895065308, accuracy: 0.4749999940395355
filters first ver. adding Dropout to FC -> froze

conv2d (Conv2D)              (None, 149, 149, 32)      416
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 148, 148, 32)      4128
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 74, 74, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 73, 73, 16)        2064
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 72, 72, 16)        1040
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 36, 36, 16)        0
_________________________________________________________________
flatten (Flatten)            (None, 20736)             0
_________________________________________________________________
dense (Dense)                (None, 128)               2654336
_________________________________________________________________
dense_1 (Dense)              (None, 32)                4128
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 264
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 27
'''

'''
SEQUENTIAL still prints multiple losses and accuracy, 
unlike previous Functional API, this model didn't stuck at loss/acc.

changed model's layers to match previous Functional model
  > that is adding Dropout after each Conv layer and switching Flatten -> GlobalAVG
  > model froze

removed all the Dropout
  > no changes

exchange GAP to Flatten
  > no changes

MaxPool arg: (2, 2)
  > no changes

the # of nodes, filters, units adjusted
  > worked
  > ***the complexity should match that of desired outcome***

Still not quiet getting why the result gives multiple losses and accuracies.
Perhaps it's for layers' multiplicity.
'''