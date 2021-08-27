# Building functional AutoEncoder

import random
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255


def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model


model = autoencoder(hidden_layer_size=154)
# PCA 95%

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, x_train, epochs=12, batch_size=128, validation_split=0.15)

decoded_img = model.predict(x_test)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(20, 7))
'''
Returns
 > fig : ~.figure.Figure
 > ax : .axes.Axes or array of Axes
        *ax* can be either a single ~matplotlib.axes.Axes object or an array of Axes objects if more than one subplot was created. 
        The dimensions of the resulting array can be controlled with the squeeze keyword.
'''
random_image = random.sample(range(decoded_img.shape[0]), 5)

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_image[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('Input', size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(decoded_img[random_image[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('Output', size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
# Adjust the padding between and around subplots.
plt.show()