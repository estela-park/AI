# Without train/test split
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
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

xy_data = datagen.flow_from_directory(
    '../_data/sex',
    target_size=(150, 150),
    batch_size=2800,
    class_mode='binary',
    shuffle=True
)
# Found 3309 images belonging to 2 classes.
# (2800, 150, 150, 3) (2800, ) [0. 1.]
# (509, 150, 150, 3) (509, ) [0. 1.]

np.save('../_save/_npy/keras59_7_x_train.npy', arr=xy_data[0][0])
np.save('../_save/_npy/keras59_7_y_train.npy', arr=xy_data[0][1])
np.save('../_save/_npy/keras59_7_x_test.npy', arr=xy_data[1][0])
np.save('../_save/_npy/keras59_7_y_test.npy', arr=xy_data[1][1])


############################ Tearing up the structue ############################
# type(xy_train): <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# len(xy_train): 2
# type(xy_train[0]): <class 'tuple'>
# len(xy_train[0]): 2
# type(xy_train[0][0]): <class 'numpy.ndarray'>
# len(xy_train[0][0]): 2200
# type(xy_train[0][1]): <class 'numpy.ndarray'>
# len(xy_train[0][1]): 2200
# type(xy_train[1][0]): <class 'numpy.ndarray'>
# len(xy_train[1][0]): 1109
# type(xy_train[1][1]): <class 'numpy.ndarray'>
# len(xy_train[1][1]):1109
# type(xy_train[0][0][1]): <class 'numpy.ndarray'>
# xy_train[0][0][1].shape: (150, 150, 3)
# type(xy_train[0][1][1]): <class 'numpy.float32'>
# type(xy_train[1][0][1]): <class 'numpy.ndarray'>
# xy_train[1][0][1].shape: (150, 150, 3)
# type(xy_train[1][1][1]): <class 'numpy.float32'>
# np.max(xy_train[1][0][1][:][75][1]): 0.30274934
#                                      scaled from [0, 255] to [0, 1]
# np.average(xy_train[1][0][1][:][75][1]): 0.5832799