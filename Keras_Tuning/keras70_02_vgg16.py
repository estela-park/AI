# How to merge models

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16
from tensorflow.python.keras.layers.core import Flatten

vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(100, 100, 3))

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# model.trainable=False: freeze the training, 훈련 동결