import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

datasets = load_iris()
# datasets.keys(): ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename']
# datasets.target_names: ['setosa' 'versicolor' 'virginica']

x = datasets.data
# (150, 4)
y = datasets.target
# (150,)

# df = pd.DataFrame(x, columns=datasets.feature_names): works the same
df = pd.DataFrame(x, columns=datasets['feature_names'])

# Augmenting the label
df['target'] = y

sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()

#########################################CorrelationCoefficient#########################################
# A heat map is a two-dimensional representation of data in which values are represented by colors
#                                               <<Table>>
#                    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)    target
# sepal length (cm)           1.000000         -0.117570           0.871754          0.817941  0.782561
# sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126 -0.426658
# petal length (cm)           0.871754         -0.428440           1.000000          0.962865  0.949035
# petal width (cm)            0.817941         -0.366126           0.962865          1.000000  0.956547
# target                      0.782561         -0.426658           0.949035          0.956547  1.000000
#
# df.corr hasn't go through training, its credibility is somewhat arguable.