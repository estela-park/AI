import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 1e-1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 1e-1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, 0, 1)
x_test_noised = np.clip(x_test_noised, 0, 1)

'''
clips(limit) the values in an array
  > syntax: (a, a_min, a_max)
  > Given an interval, values outside the interval are clipped to the interval edges. 
    For example, if an interval of [0, 1] is specified, 
    values smaller than 0 become 0, and values larger than 1 become 1.
  > Parameters
    a : array_like
        Array containing elements to clip.
    a_min : scalar or array_like or None
        Minimum value. If None, clipping is not performed on lower interval edge. 
    a_max : scalar or array_like or None
        Maximum value. If None, clipping is not performed on upper interval edge. 
  > Returns clipped_array in ndarray
'''

model = Sequential()
model.add(Dense(units=114, input_shape=(784,), activation='relu'))
model.add(Dense(units=784, activation='sigmoid'))

model.compile(optimizer='adam', loss='mse')
model.fit(x_train_noised, x_train_noised, epochs=20, batch_size=784)
output = model.predict(x_test_noised)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20, 7))

image_num = random.sample(range(10000), 5)

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[image_num[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('Intact', size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[image_num[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('Noised-Input', size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[image_num[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('Output', size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()

# training with y being intact results in clear image
# training with y being noised results in blunter image
# > model trained with noised image encodes clearer image than the noised on