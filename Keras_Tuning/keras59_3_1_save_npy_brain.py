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
    '../_data/brain/train/',
    target_size=(150, 150),
    batch_size=200,
    class_mode='binary',
)
# Found 160 images belonging to 2 classes.
# (160, 150, 150, 3) (160, )


xy_test = test_datagen.flow_from_directory(
    '../_data/brain/test',
    target_size=(150, 150),
    batch_size=200,
    class_mode='binary',
    shuffle=True,
)
# Found 120 images belonging to 2 classes.
# (120, 150, 150, 3) (120, )

np.save('../_save/_npy/keras59_3_x_train', arr=xy_train[0][0])
np.save('../_save/_npy/keras59_3_y_train', arr=xy_train[0][1])
np.save('../_save/_npy/keras59_3_x_test', arr=xy_test[0][0])
np.save('../_save/_npy/keras59_3_y_test', arr=xy_test[0][1])