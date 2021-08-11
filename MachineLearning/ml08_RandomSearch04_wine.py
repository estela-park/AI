import time
from sklearn.datasets import load_wine
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


# 1: Data-prep
dataset = load_wine()

x = dataset.data   
y = dataset.target 

kfold = KFold(n_splits=5)

parameters = [
    {'n_jobs': [-1], 'n_estimators': [100, 200], 'max_depth': [6, 8, 10], 'min_samples_leaf': [5, 7, 10]},
    {'n_jobs': [-1], 'max_depth': [5, 6, 7, 9], 'min_samples_leaf': [3, 6, 9, 11],  'min_samples_split': [3, 4, 5]},
    {'n_jobs': [-1], 'min_samples_leaf': [3, 5, 7],  'min_samples_split': [2, 3, 5, 10]},
    {'n_jobs': [-1], 'min_samples_split': [2, 3, 5, 10]},
]

# 2: Modelling
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1)
model_r = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1)


# 3: Training; GridSearchCV supports fitting
start = time.time()
model.fit(x, y)
end = time.time() - start
start_r = time.time()
model_r.fit(x, y)
end_r = time.time() - start_r


# 4: Evaluation
print('GridSearch took', end, 'seconds')
print('RandomSearch took', end_r, 'seconds')
print('best esimator was', model.best_estimator_)
print('best parameter was', model.best_params_)
print('Random_best estimator was', model_r.best_estimator_)
print('Random_best parameter was', model_r.best_params_)
print('score:',model.score(x, y))
print('Random_score:',model_r.score(x, y))

'''
Fitting 5 folds for each of 82 candidates, totalling 410 fits
Fitting 5 folds for each of 10 candidates, totalling 50 fits
GridSearch took 35.30908226966858 seconds
RandomSearch took 3.883607864379883 seconds
best esimator was RandomForestClassifier(n_jobs=-1)
best parameter was {'min_samples_split': 2, 'n_jobs': -1}
Random_best estimator was RandomForestClassifier(min_samples_leaf=3, min_samples_split=10, n_jobs=-1)
Random_best parameter was {'n_jobs': -1, 'min_samples_split': 10, 'min_samples_leaf': 3}
score: 1.0
Random_score: 1.0
'''