import numpy as np

x = np.array([range(100), range(301, 401), range(1, 101), range(100), range(401, 501)])
x = np.transpose(x)
y = np.array([range(711, 811), range(101, 201)])
y = np.transpose(y)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

                           # model = Sequential()
input1 = Input(shape=(5,)) # model.add(Dense(3, input_shape=(5, ))), for Model function, input layer should be specified
dense1 = Dense(3)(input1)  # for Model function, every layer should be initialized
dense2 = Dense(4)(dense1)  # model.add(Dense(4))
dense3 = Dense(10)(dense2) # model.add(Dense(10))
output1 = Dense(2)(dense3) # model.add(Dense(2))

model = Model(inputs=input1, outputs=output1)
                           # Model can be paralleled, that is models can interact, merge, expand etc. it's called ensemble.
model.summary()