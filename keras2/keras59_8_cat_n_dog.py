import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    '../_data/catNdog/train',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    shuffle=True,
)
# Found 8005 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/catNdog/test',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    shuffle=True,
)
# Found 2023 images belonging to 2 classes.

start = time.time()
# model.fit(xy_train[:][0], xy_train[:][1])

hist = model.fit_generator(xy_train, epochs=24, steps_per_epoch=32, validation_data=xy_test,
                           # validation_steps=4: doesn't work solo w/o validation_data
                           # batch_size is unexpected as generator already deals with it.
)
# steps_per_epoch=the number of images/batch<as specified in generator>                   
end = time.time() - start