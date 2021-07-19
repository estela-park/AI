from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras.layers.core import Flatten

model = Sequential()
model.add(Conv2D(10, kernel_size=(2, 2), input_shape=(5, 5, 1))) # (N, 4, 4, 10), N = None
model.add(Conv2D(20, (2,2), activation='relu'))
model.add(Conv2D(20, (2,2), activation='relu'))
model.add(Flatten())                                             # (N, 160)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()