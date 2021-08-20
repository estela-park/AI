import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)
# Hyperbolic functions are analogues of the ordinary trigonometric functions, 
# but defined using the hyperbola rather than the circle. 
# tanh(x) = sinh(x) / cosh(x) = (e^2x − 1) / (e^2x + 1)
# sinh(x) = (e^x − e^−x) / 2
# cosh(x) = (e^x + e^−x) / 2
plt.plot(x, y)
plt.grid()
plt.show()