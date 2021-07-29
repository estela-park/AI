import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.2,
    shear_range=0.5,
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(rescale=1./255)

augment_size = 40000

randidx = np.random.randint(x_train.shape[0], size=augment_size)

# x_train[0].shape: (28, 28)
# x_train.shape[0]: 60000
# randidx: [35025 58535 14977 ...  4704  5410 32745]
# randidx.shape: (40000,)

x_augmented = x_train[randidx].copy().reshape(40000, 28, 28, 1)
# (40000, 28, 28)
y_augmented = y_train[randidx].copy()
# .copy() prevent sharing memory address
# using same [randidx] makes sure x and y are coupled.

# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_train.reshape(x_test.shape[0], 28, 28, 1)

print(x_augmented.shape)
print(x_train.shape)
print(x_test.shape)

x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size), batch_size=augment_size, shuffle=False
                                 ,save_to_dir='../temp')[0]

# x_train = np.concatenate((x_train, x_augmented))
# y_train = np.concatenate((y_train, y_augmented))