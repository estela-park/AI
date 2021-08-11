import time
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# 1. Data-prep
dataset = load_iris()
x = dataset.data   
y = dataset.target 

kfold = KFold(n_splits=5)


# 2. Modelling
# Pipeline can gives arg alias in tuple
pipe = Pipeline([('scaler', MinMaxScaler()), ('rf', RandomForestClassifier())])

# model name should be given in alias
parameters = [
    {'rf__n_jobs': [-1], 'rf__n_estimators': [100, 200], 'rf__max_depth': [6, 8, 10], 'rf__min_samples_leaf': [5, 7, 10]},
    {'rf__n_jobs': [-1], 'rf__max_depth': [5, 6, 7, 9], 'rf__min_samples_leaf': [3, 6, 9, 11],  'rf__min_samples_split': [3, 4, 5]},
    {'rf__n_jobs': [-1], 'rf__min_samples_leaf': [3, 5, 7],  'rf__min_samples_split': [2, 3, 5, 10]},
    {'rf__n_jobs': [-1], 'rf__min_samples_split': [2, 3, 5, 10]},
]
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
GridSearch took 5.059211730957031 seconds
best esimator was Pipeline(steps=MinMaxScaler(), RandomForestClassifier(max_depth=10, min_samples_leaf=5, n_jobs=-1))])
best parameter was {'rf__n_jobs': -1, 'rf__n_estimators': 100, 'rf__min_samples_leaf': 5, 'rf__max_depth': 10}
score: 0.9733333333333334
'''