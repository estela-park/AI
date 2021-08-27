import random
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, UpSampling2D

(x_train, _), (x_test, _) = mnist.load_data()
train_x = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
train_y = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.


def ae_CNN_vanilla(hidden_layer):
    model = Sequential()
    model.add(Conv2D(hidden_layer, (2, 2), input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(hidden_layer//4, (2, 2), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPool2D((3, 3)))
    model.add(Flatten())
    model.add(Dense(784, activation='sigmoid'))
    return model


def ae_CNN_deep(hidden_layer):
    model = Sequential()
    model.add(Conv2D(hidden_layer, (2, 2), input_shape=(28, 28, 1), activation='relu'))
    model.add(UpSampling2D())
              # default: size=(2, 2)
              # size=(a, b) 
              # > horizontally a times
              # > vertically b times
    model.add(MaxPool2D((3, 3)))
    model.add(Conv2D(28, (2, 2), activation='relu'))
    model.add(UpSampling2D())
    model.add(MaxPool2D())
    model.add(Conv2D(14, (2, 2), activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(784, activation='sigmoid'))
    return model


model_vanilla = ae_CNN_vanilla(56)
model_deep = ae_CNN_deep(56)
model_vanilla.summary()
model_deep.summary()
model_vanilla.compile(optimizer='adam', loss='mse')
model_deep.compile(optimizer='adam', loss='mse')

model_vanilla.fit(train_x, train_y, epochs=12, batch_size=64)
model_deep.fit(train_x, train_y, epochs=12, batch_size=64)

output_vanilla = model_vanilla.predict(x_test)
output_deep = model_deep.predict(x_test)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20, 7))

image_num = random.sample(range(10000), 5)

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[image_num[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('Input', size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output_vanilla[image_num[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('Vanilla_Output', size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output_deep[image_num[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('Deep_Output', size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()