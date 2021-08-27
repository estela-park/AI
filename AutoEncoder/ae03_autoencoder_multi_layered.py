import random
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.


def ae_vanilla(hidden_layer):
    model = Sequential()
    model.add(Dense(hidden_layer, input_shape=(784,), activation='relu'))
    model.add(Dense(784, activation='selu'))
    return model


def ae_deep(hidden_layer):
    model = Sequential()
    model.add(Dense(hidden_layer, input_shape=(784,), activation='relu'))
    model.add(Dense(1568, activation='relu'))
    # activation applied on hidden layers doesn't do much for weight optimizing,
    # rather, it limit the spectrum of output_value, thus how the image looks.
    model.add(Dense(3136, activation='relu'))
    model.add(Dense(1568, activation='relu'))
    model.add(Dense(784, activation='selu'))
    return model


model_vanilla = ae_vanilla(64)
model_deep = ae_deep(64)

model_vanilla.compile(optimizer='adam', loss='mse')
model_deep.compile(optimizer='adam', loss='mse')

model_vanilla.fit(x_train, x_train, epochs=20, batch_size=784)
model_deep.fit(x_train, x_train, epochs=20, batch_size=784)

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

'''
selu
 > vanilla: 뿌옇다
 > deep: 검은 반점이 찍혔다
relu, output: selu
 > vanilla: 엄청 blurred
 > deep: 검은 반점+all selu 유사
relu, output: sigmoid
 > vanilla: 약간 blurred
 > deep: 근소하게 또렸하다
'''