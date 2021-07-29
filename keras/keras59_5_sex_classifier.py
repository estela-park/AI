import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

x_train = np.load('../_save/_npy/k59_3_train_x.npy')
# <class 'numpy.ndarray'> (2200, 150, 150, 3)
y_train = np.load('../_save/_npy/k59_3_train_y.npy')
# <class 'numpy.ndarray'> (2200,)
x_test = np.load('../_save/_npy/k59_3_test_x.npy')
# <class 'numpy.ndarray'> (1109, 150, 150, 3)
y_test = np.load('../_save/_npy/k59_3_test_y.npy')
# <class 'numpy.ndarray'> (1109,)

test_datagen = ImageDataGenerator(rescale=1./255)

predict = test_datagen.flow_from_directory(
    '../_data/sex_test',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    shuffle=True,
)



model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='same',                          
                        activation='relu', input_shape=(150, 150, 3))) 
model.add(Dropout(0.35))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation="sigmoid"))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=8, verbose=2)

start = time.time()
hist = model.fit(x_train, y_train, epochs=64, steps_per_epoch=35,
                 validation_data=(x_test, y_test), validation_steps=18,
                 # steps_per_epoch & validation_steps is optional, 
                 # when these are set to default, fitting takes longer time.
                 verbose=2,callbacks=[es],)
end = time.time() - start


# 0: men, 1: women
prediction = model.predict(predict)
print(prediction)
print('time:',end)

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 150, 150, 64)      3136
_________________________________________________________________
dropout (Dropout)            (None, 150, 150, 64)      0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 150, 150, 64)      65600
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 75, 75, 64)        0
_________________________________________________________________
global_average_pooling2d (Gl (None, 64)                0
_________________________________________________________________
dense (Dense)                (None, 128)               8320
_________________________________________________________________
dense_1 (Dense)              (None, 32)                4128
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 264
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 9
=================================================================
Total params: 81,457
'''
