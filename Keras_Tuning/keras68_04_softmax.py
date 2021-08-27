import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
    # exponential works as amplification of signal
    # / np.sum makes sure that the sub of softmax's products equals to 1.

x = np.arange(1, 5)
y = softmax(x)

plt.pie(y, labels=y, shadow=True, startangle=90)
plt.show()