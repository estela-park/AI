# how iterator works

import time
from tensorflow.keras.datasets import fashion_mnist
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

test_datagen = ImageDataGenerator(rescale=1./255)

(x_train, _), _ = fashion_mnist.load_data() 
# (60000, 28, 28) 
x_train = x_train.reshape(-1, 28, 28, 1)

x_aug = datagen.flow(x_train, batch_size=16, save_to_dir='../temp', save_prefix=str(time.time()))

time.sleep(3)
print(x_aug.next())
print('first')

time.sleep(3)
print(x_aug.next())
print('second')

time.sleep(3)
print(x_aug.next())
print('third')