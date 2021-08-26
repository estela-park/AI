import numpy as np
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

# flow_from_directory: every image in directory is loaded.
# there are sub-directories of which names are used as labels.
xy_train = train_datagen.flow_from_directory(
    '../_data/brain/train/',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    # default: shuffle=True,
    # default: color_mode='rgb'
)
# Found 160 images belonging to 2 classes.
# xy_train                                                                              : DirectoryIterator
# xy_train[0 ~ # of images/batch]                                                       : Tuple
# x: xy_train[0 ~ # of images/batch][0]: (5, 150, 150, 3): batch, pixcels, color        : ndarray
# y: xy_train[0 ~ # of images/batch][1]: (5, ): batch, binary=scalar|categorical=vector : ndarray

xy_test = test_datagen.flow_from_directory(
    '../_data/brain/test',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    shuffle=True,
)
# Found 120 images belonging to 2 classes.