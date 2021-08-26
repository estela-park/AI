import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28), (60000,), (10000, 28, 28), (10000,)
                                                         # x_train[0]: 28x28 metrics with entries 0~255
                                                         # y_train[0]: 5
plt.imshow(x_train[0], 'gray')
plt.show()