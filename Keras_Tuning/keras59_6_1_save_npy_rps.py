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
    '../_data/rps',
    target_size=(150, 150),
    batch_size=2000,
    class_mode='categorical',
    shuffle=True
)
# Found 2520 images belonging to 3 classes.
# (2000, 150, 150, 3) (2000, 3)_one-hot encoded
# (520, 150, 150, 3) (520, 3)_one-hot encoded

np.save('../_save/_npy/keras59_6_x_train', arr=xy_data[0][0])
np.save('../_save/_npy/keras59_6_y_train', arr=xy_data[0][1])
np.save('../_save/_npy/keras59_6_x_test', arr=xy_data[1][0])
np.save('../_save/_npy/keras59_6_y_test', arr=xy_data[1][1])