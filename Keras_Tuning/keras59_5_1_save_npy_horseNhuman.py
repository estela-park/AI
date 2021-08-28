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
    '../_data/horseNhuman/train/',
    target_size=(150, 150),
    batch_size=800,
    class_mode='binary',
)
# Found 1027 images belonging to 2 classes.
# (800, 150, 150, 3) (800,)

xy_test = test_datagen.flow_from_directory(
    '../_data/horseNhuman/test',
    target_size=(150, 150),
    batch_size=800,
    class_mode='binary',
    shuffle=True,
)
# Found 256 images belonging to 2 classes.
# (256, 150, 150, 3) (256,)

np.save('../_save/_npy/keras59_5_x_train', arr=xy_train[0][0])
np.save('../_save/_npy/keras59_5_y_train', arr=xy_train[0][1])
np.save('../_save/_npy/keras59_5_x_test', arr=xy_test[0][0])
np.save('../_save/_npy/keras59_5_y_test', arr=xy_test[0][1])