import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# 1: Data-prep
dataset = load_breast_cancer()

x = dataset.data   
y = dataset.target 

kfold = KFold(n_splits=5)


# 2: Modelling
model = LinearSVC()


# 3: Training
#            > not needed


# 4: Evaluation
scores = cross_val_score(model, x, y, cv=kfold)
print('model: LinearSVC, k-value: 5, accuracy:', scores, round(np.mean(scores), 4))

'''
model: LinearSVC,               k-value: 5, accuracy: [0.83333333 0.92105263 0.92105263 0.92982456 0.9380531 ] 0.9087
model: SVC,                     k-value: 5, accuracy: [0.77192982 0.90350877 0.97368421 0.93859649 0.94690265] 0.9069
model: KNeighborsClassifier,    k-value: 5, accuracy: [0.85964912 0.92105263 0.96491228 0.94736842 0.9380531 ] 0.9262
model: LogisticRegression,      k-value: 5, accuracy: [0.9122807  0.92982456 0.97368421 0.92982456 0.95575221] 0.9403
model: DecisionTreeClassifier,  k-value: 5, accuracy: [0.88596491 0.92982456 0.95614035 0.92982456 0.84955752] 0.9103
model: RandomForestClassifier,  k-value: 5, accuracy: [0.9122807  0.93859649 0.98245614 0.96491228 0.98230088] 0.9561
'''