import time
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline


# 1. Data-prep
dataset = load_diabetes()
x = dataset.data   
y = dataset.target 

kfold = KFold(n_splits=5)

parameters = [
    {'randomforestregressor__n_jobs': [-1], 'randomforestregressor__n_estimators': [100, 200], 'randomforestregressor__max_depth': [6, 8, 10], 'randomforestregressor__min_samples_leaf': [5, 7, 10]},
    {'randomforestregressor__n_jobs': [-1], 'randomforestregressor__max_depth': [5, 6, 7, 9], 'randomforestregressor__min_samples_leaf': [3, 6, 9, 11],  'randomforestregressor__min_samples_split': [3, 4, 5]},
    {'randomforestregressor__n_jobs': [-1], 'randomforestregressor__min_samples_leaf': [3, 5, 7],  'randomforestregressor__min_samples_split': [2, 3, 5, 10]},
    {'randomforestregressor__n_jobs': [-1], 'randomforestregressor__min_samples_split': [2, 3, 5, 10]},
]


# 2. Modelling
pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor())
model = RandomizedSearchCV(pipe, parameters, cv=kfold, verbose=1)


# 3. Training
start = time.time()
model.fit(x, y)
end = time.time() - start


# 4. Evaluation
print('GridSearch took', end, 'seconds')
print('best esimator was', model.best_estimator_)
print('best parameter was', model.best_params_)
print('score:',model.score(x, y))

'''
Fitting 5 folds for each of 10 candidates, totalling 50 fits
GridSearch took 5.151446580886841 seconds
best esimator was Pipeline(steps=MinMaxScaler(), RandomForestRegressor(max_depth=6, min_samples_leaf=6, min_samples_split=5, n_jobs=-1))])
best parameter was {'randomforestregressor__n_jobs': -1, 'randomforestregressor__min_samples_split': 5, 'randomforestregressor__min_samples_leaf': 6, 'randomforestregressor__max_depth': 6}
score: 0.7016061406704566
'''