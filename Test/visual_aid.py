import numpy as np
transformer = np.eye(4)
# transformer[0,0] = 'scale'
# transformer[1,1] = 'scale'
# transformer[2,2] = 'scale'
transformer[:, 3] = -1
print(transformer)