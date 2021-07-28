import numpy as np

x_train = np.load('../_save/_npy/k59_3_train_x.npy')
# <class 'numpy.ndarray'> (2200, 150, 150, 3)
y_train = np.load('../_save/_npy/k59_3_train_y.npy')
# <class 'numpy.ndarray'> (2200,)
x_test = np.load('../_save/_npy/k59_3_test_x.npy')
# <class 'numpy.ndarray'> (1109, 150, 150, 3)
y_test = np.load('../_save/_npy/k59_3_test_y.npy')
# <class 'numpy.ndarray'> (1109,)