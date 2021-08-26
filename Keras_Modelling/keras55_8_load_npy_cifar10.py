import numpy as np

x_data = np.load('./_save/_npy/k55_x_data_cifar10.npy')
y_data = np.load('./_save/_npy/k55_y_data_cifar10.npy')

(x_train, y_train), (x_test, y_test) = (x_data[0], y_data[0]), (x_data[1], y_data[1])