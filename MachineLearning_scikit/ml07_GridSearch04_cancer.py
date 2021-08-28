import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# 1: Data-prep
dataset = load_breast_cancer()

x = dataset.data   
y = dataset.target 

kfold = KFold(n_splits=5)



# 2. Modelling with GridSearch
parameters = [
    {'n_jobs': [-1], 'n_estimators': [100, 200], 'max_depth': [6, 8, 10], 'min_samples_leaf': [5, 7, 10]},
    {'n_jobs': [-1], 'max_depth': [5, 6, 7, 9], 'min_samples_leaf': [3, 6, 9, 11],  'min_samples_split': [3, 4, 5]},
    {'n_jobs': [-1], 'min_samples_leaf': [3, 5, 7],  'min_samples_split': [2, 3, 5, 10]},
    {'n_jobs': [-1], 'min_samples_split': [2, 3, 5, 10]},
]

model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold)



# 3: Training; GridSearchCV supports fitting
model.fit(x, y)



# 4: Tuning with best estimator
print('best parameter was', model.best_estimator_)
print('score:',model.score(x, y))

model = RandomForestClassifier(max_depth=9, min_samples_leaf=3, min_samples_split=4, n_jobs=-1)
model.fit(x, y)

print('model was set to max_depth=9, min_samples_leaf=3, min_samples_split=4, n_jobs=-1')
print('score:',model.score(x, y))

'''
best parameter was RandomForestClassifier(max_depth=5, min_samples_leaf=3, min_samples_split=5, n_jobs=-1)
score: 0.9894551845342706
model was set to max_depth=9, min_samples_leaf=3, min_samples_split=4, n_jobs=-1
score: 0.9894551845342706
'''