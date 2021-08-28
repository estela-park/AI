import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data()
# (6000, 28, 28) (10000, 28, 28)

# There are 5 cases for using the underscore(_) in Python.
# 
# > 1. For storing the value of last expression in interpreter.
# > 2. For ignoring the specific values. (so-called “I don’t care”)
# > 3. To give special meanings and functions to name of vartiables or functions.
# > 4. To use as ‘Internationalization(i18n)’ or ‘Localization(l10n)’ functions.
# > 5. To separate the digits of number literal value.

x = np.append(x_train, x_test, axis=0)
# (70000, 28, 28)

# Since PCA operates on the premise of linear correlation, features should be aligned in 1D.
x = x.reshape(70000, 28*28)

pca = PCA()
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_
np.argmax(np.cumsum(pca_EVR) >= 0.95)

# default: 784
# 1 :      713
# 0.9:     87
# 0.95:    154
# 0.99:    331
# 0.999:   486
