import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC


# 1: Data-prep
dataset = load_iris()

x = dataset.data   
y = dataset.target 

kfold = KFold(n_splits=5)

parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
    {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]},
]

# 2: Modelling
model = SVC(C=1, kernel='linear')
# it runs (the number of parameters X K) times.


# 3: Training; GridSearchCV supports fitting
model.fit(x, y)


# 4: Evaluation
print('model was set to C=1, kernel="linear"')
print('score:',model.score(x, y))

'''
model was set to C=1, kernel="linear"
score: 0.9933333333333333
'''