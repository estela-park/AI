import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.
y_intact = x_train.reshape(60000, 784).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 1e-1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 1e-1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, 0, 1)
x_test_noised = np.clip(x_test_noised, 0, 1)
y_noised = x_train_noised.reshape(60000, 784)

model = Sequential()
model.add(Conv2D(116, (2, 2), input_shape=(28, 28, 1), activation='relu'))
model.add(Dropout(0.15))
model.add(Conv2D(58, (2, 2), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(29, (2, 2), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(units=784, activation='sigmoid'))

model.compile(optimizer='adam', loss='mse')
model.fit(x_train_noised, y_intact, epochs=12, batch_size=784)
model.fit(x_train_noised, y_noised, epochs=12, batch_size=784)
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