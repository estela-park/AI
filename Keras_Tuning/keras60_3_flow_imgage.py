# Visualize what ImageDataGenerator.flow() does

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

(x_train, _), _ = fashion_mnist.load_data()
# (60000, 28, 28)

augment_size = 40000
random_idx = np.random.randint(x_train.shape[0], size=augment_size)

x_aug = x_train[random_idx].copy().reshape(augment_size, 28, 28, 1)
x_train = x_train[random_idx].copy().reshape(augment_size, 28, 28, 1)
x_aug = datagen.flow(x_aug, batch_size=augment_size, shuffle=False).next()

plt.figure(figsize=(2, 10))
for i in range(20):
    if i <= 9:
        plt.subplot(2, 10, i+1)
        plt.axis('off')
        plt.imshow(x_train[i], cmap='gray')
    else:
        plt.subplot(2, 10, i+1)
        plt.axis('off')
        plt.imshow(x_aug[i-10], cmap='gray')
plt.show()