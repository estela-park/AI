import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense

x_train = np.load('../_save/_npy/keras59_7_x_train.npy')
# (2800, 150, 150, 3)
x_test = np.load('../_save/_npy/keras59_7_x_test.npy')
# (509, 150, 150, 3)


y_intact = x_train.reshape(2800, 150*150*3)

x_train_noised = x_train + np.random.normal(0, 5e-2, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 5e-2, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, 0, 1)
x_test_noised = np.clip(x_test_noised, 0, 1)
y_noised = x_train_noised.reshape(2800, 150*150*3)

model = Sequential()
model.add(Conv2D(60, (2, 2), input_shape=(150, 150, 3), activation='relu'))
model.add(Dropout(0.15))
model.add(Conv2D(30, (2, 2), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(15, (2, 2), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.35))
model.add(Dense(2024))
model.add(Dropout(0.15))
model.add(Dense(units=150*150*3, activation='sigmoid'))

model.compile(optimizer='adam', loss='mse')
# model.fit(x_train, y_noised, epochs=12, batch_size=64)
model.fit(x_train_noised, y_noised, epochs=12, batch_size=64)
# model.fit(x_train_noised, y_intact, epochs=12, batch_size=64)
output = model.predict(x_test_noised)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20, 7))

image_num = random.sample(range(509), 5)

# (M, N): an image with scalar data. 
#         The values are mapped to colors using normalization and a colormap. 
#         Takes parameters norm, cmap, vmin, vmax.
# (M, N, 3): an image with RGB values (0-1 float or 0-255 int).
# (M, N, 4): an image with RGBA values (0-1 float or 0-255 int), i.e. including transparency

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[image_num[i]].reshape(150, 150, 3))
    if i == 0:
        ax.set_ylabel('Intact', size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[image_num[i]].reshape(150, 150, 3))
    if i == 0:
        ax.set_ylabel('Noised-Input', size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[image_num[i]].reshape(150, 150, 3))
    if i == 0:
        ax.set_ylabel('Output', size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()