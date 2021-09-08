import numpy as np
points = np.array([[['X1', 'X2', 'X3', 'X4'], ['Y1', 'Y2', 'Y3', 'Y4'], ['Z1', 'Z2', 'Z3', 'Z4']]])
print(points)
print(points.shape)
points = np.repeat(points, 1, axis=0)
print(points)
print(points.shape)