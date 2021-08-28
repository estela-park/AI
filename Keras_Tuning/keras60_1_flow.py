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

augment_size = 5
x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),
    np.zeros(augment_size),
    batch_size=augment_size,
    shuffle=False,
)
# ImageDataGenerator.flow(x, y, batchz_size=)
# x_data                             : NumpyArrayIterator       :.next()_Tuple
#   > executing .next()_             : x_data[0]: x    /   x_data[1]: label
#                                      (batch, 28, 28, 1)  (batch, )
# x_data[0 ~ # of images / batch]    : Tuple  : train_data_set
# x_data[0 ~ # of images / batch][0] : ndarray: x data
# x_data[0 ~ # of images / batch][1] : ndarray: label


plt.figure(figsize=(7, 7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    print(x_data[0][i])
    print(x_data[0][i].shape)
    plt.imshow(x_data[0][i], cmap='gray')
# plt.imshow expects image data as [height, width, 3]

plt.show()