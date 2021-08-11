import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
'''
GridSearchCV:
   one kind of model which has adjustable parameters in the form of dictionary.
   CV in its name suggests that it also does cross-validation
'''
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
model = GridSearchCV(SVC(), parameters, cv=kfold)
# it runs (the number of parameters X K) times.


# 3: Training; GridSearchCV supports fitting
model.fit(x, y)


# 4: Evaluation
print('best parameter was', model.best_estimator_)
print('score:',model.score(x, y))

'''
best parameter was SVC(C=1, kernel='linear')
score: 0.9933333333333333
'''