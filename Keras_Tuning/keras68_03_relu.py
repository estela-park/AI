import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# elu
#  >  if x>0,   x
#  >  if x≤0,   α∗(exp(x)−1) it converges to zero as x goes -inf.
#     > alpha: the value for the ELU formulation. Default: 1.0
#
# selu: smoothness
#  >  if x>0,   x
#  >  if x≤0,   α∗(e^x − 1)   
# > In contrast to ReLUs, SELUs doesn't die.
# > SELU has smooth line, it works better with non-linear problem.
#
# leaky relu
#  >  if x>0,   x
#  >  if x≤0,   αx
#     > alpha	Float >= 0. slope coefficient. Default to 0.3.