# EllipticEnvelope 원리 찾아서 정리해놓기

# Assuming that the data comes from a known distribution (e.g. data are Gaussian distributed), 
# data should fit an ellipse to the central data points.
# The inlier data are Gaussian distributed.

import numpy as np
from sklearn.covariance import EllipticEnvelope

aaa = np.array([[1, 2, 10000, 3, 4, 6, 7, 8, 90, 100],
                 [1000, 2000, 3, 4000, 5000, 6000, 7000, 8, 9000, 10000]]).transpose()
# (10, 2)

outliers = EllipticEnvelope(contamination=.2)
# Anormal data is called to be contaminated.
# contamination=0.1: the proportion of outliers in the data set. Range is (0, 0.5)
outliers.fit(aaa)
print(outliers.predict(aaa))

# 1: inlier, -1: outlier
#  > [ 1  1 -1  1  1  1  1  1  1 -1]