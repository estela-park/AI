import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../data/brain//train/',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
)


model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(150, 150, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary-crossentropy', optimizer='adam', metrics=['acc'])

start = time.time()
# model.fit(xy_train[:][0], xy_train[:][1])
hist = model.fit_generator(xy_train, epochs=24, steps_per_epoch=32, validation_data=xy_test,
                           # validation_steps=4: doesn't work solo w/o validation_data
)
# steps_per_epoch=the number of images/batch<as specified in generator>                   
end = time.time() - start

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']