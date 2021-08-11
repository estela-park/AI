import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# 1: Data-prep
dataset = load_wine()

x = dataset.data   
y = dataset.target 

kfold = KFold(n_splits=5)


# 2: Modelling
model = RandomForestClassifier()


# 3: Training
#            > not needed


# 4: Evaluation
scores = cross_val_score(model, x, y, cv=kfold)
print('model: RandomForestClassifier, k-value: 5, accuracy:', scores, round(np.mean(scores), 4))

'''
model: LinearSVC,               k-value: 5, accuracy: [0.61111111 0.75       1.         0.74285714 0.05714286] 0.6322
model: SVC,                     k-value: 5, accuracy: [0.83333333 0.75       0.5        0.62857143 0.        ] 0.5424
model: KNeighborsClassifier,    k-value: 5, accuracy: [0.86111111 0.83333333 0.55555556 0.77142857 0.02857143] 0.61
model: LogisticRegression,      k-value: 5, accuracy: [0.91666667 0.88888889 0.88888889 0.91428571 1.        ] 0.9217
model: DecisionTreeClassifier,  k-value: 5, accuracy: [0.91666667 0.83333333 0.77777778 0.68571429 0.94285714] 0.8313
model: RandomForestClassifier,  k-value: 5, accuracy: [0.94444444 0.86111111 0.94444444 1.         0.97142857] 0.9443
'''