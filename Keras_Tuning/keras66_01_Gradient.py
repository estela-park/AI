import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 -4*x + 6
x = np.linspace(-1, 6, 100)
# from arg0 to arg1, cut it into 100 pieces
y = f(x)

plt.plot(x, y, 'k-')
# 'k'	black
plt.plot(2, f(2), 'sk')
# 's'	square marker
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')

plt.show()

'''
'-'	solid line style
'--'	dashed line style
'-.'	dash-dot line style
':'	dotted line style
'''