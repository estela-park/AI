# generated data should be combined back first, then can be saved with np.save
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

xy_train = train_datagen.flow_from_directory(
    '../_data/sex_classifier',
    target_size=(150, 150),
    batch_size=2200,
    class_mode='binary',
    shuffle=True,
)
# Found 3309 images belonging to 2 classes.

np.save('../_save/_npy/k59_3_train_x.npy', arr=xy_train[0][0])
np.save('../_save/_npy/k59_3_train_y.npy', arr=xy_train[0][1])
np.save('../_save/_npy/k59_3_test_x.npy', arr=xy_train[1][0])
np.save('../_save/_npy/k59_3_test_y.npy', arr=xy_train[1][1])