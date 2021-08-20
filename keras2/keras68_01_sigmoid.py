import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))
    # np.exp: returns the product of natural exponential fn, in either scalar or ndarray

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.grid()
# add grid in the background
plt.show()