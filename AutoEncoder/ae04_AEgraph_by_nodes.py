import random
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.


def autoencoder(hidden_layer):
    model = Sequential()
    model.add(Dense(hidden_layer, input_shape=(784,), activation='relu'))
    model.add(Dense(784, activation='sigmoid'))
    return model


images = []
for i in range(6):
    a = 2**i
    print(f'======Node {a}======')
    model = autoencoder(a)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(x_train, x_train, epochs=20, batch_size=784)
    images.append(model.predict(x_test))
    images.append(f'Node{a}')

fig, axes = plt.subplots(7, 5, figsize=(20, 7))
# axes is a ndarray which holds 7x5 spaces

image_num = random.sample(range(10000), 5)
print(len(images))
print(image_num)
for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        print(row_num, col_num)
        ax.imshow(images[(row_num-1)*2][image_num[col_num]].reshape(28, 28), cmap='gray')
        if (row_num == 0) & (col_num == 0):
            ax.set_ylabel('Input', size=20)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
        elif col_num == 0:
            ax.set_ylabel(images[(row_num-1)*2 +1], size=20)

plt.show()

# the more the nodes are, the clearer the image is