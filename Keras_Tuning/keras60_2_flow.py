import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(
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

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

augment_size = 40
random_idx = np.random.randint(x_train.shape[0], size=augment_size)
# x_train[0].shape: (28, 28)
# x_train.shape[0]: 60000
# random_idx: [35025 58535 14977 ...  4704  5410 32745]
# random_idx.shape: (40000,)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_augmented = x_train[random_idx].copy().reshape(40, 28, 28, 1)
# shuffled and multiplied w/o transformation (40, 28, 28)
y_augmented = y_train[random_idx].copy()
# .copy() prevents sharing address in the memory
# using same [random_idx] makes sure x and y are coupled.

print('============Before Iterator is called============')
print(x_augmented.shape)
print(x_train.shape)

x_augmented = datagen.flow(x_augmented, batch_size=augment_size, save_to_dir='../temp', save_prefix=f'{cnt}th').next()

# w/o next() x_augmented repeats itself infinitely; flow() is infinite loop
# but when x_augmented.next() is called, a batch is generated
x_train = np.concatenate((x_train, x_augmented))


# w/h .next()
#     > when flow() is called, that is iterator is initialized, a batch is generated
#     > since it's been haulted by .next(), 
#     > even the iterator(created by flow(), x_augmented) is called, doesn't do a thing.
# w/o .next()
#     > when flow() is called, that is iterator is initialized, notheing happens
#     > x_augmented.shape: 'NumpyArrayIterator' object has no attribute 'shape'
#     > iterator being called doesn't iterate, flow() should be iterated to generate images