# explained_variance_ratio: Rate of variance explained by each of the selected components.
# If n_components is not set then all components are stored and the sum of the ratios is equal to 1.0.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA


# 1. Data-prep
datasets = load_diabetes()
x = datasets.data
# (442, 10)
y = datasets.target
# (442, )

pca = PCA()
pca.fit(x)

pca_EVR = pca.explained_variance_ratio_
# list or ndarray

sum(pca_EVR)
# default:          1.0
# n_components=9:   0.9991439470098977
# n_components=7:   0.9479436357350414


# np.accumulate = np.cumsum: difference exists in dimensionality
np.cumsum(pca_EVR)
# default:          [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759 0.94794364 0.99131196 0.99914395 1.        ]
# n_components=9:   [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759 0.94794364 0.99131196 0.99914395]
# n_components=7:   [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759 0.94794364]

# confusing... shouldn't it be argmin?
np.argmax(np.cumsum(pca_EVR) >= 0.94)
# argmax: 6, argmin: 0

plt.plot(np.cumsum(pca_EVR))
plt.grid()
plt.show()