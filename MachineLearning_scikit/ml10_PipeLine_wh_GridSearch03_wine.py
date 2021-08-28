import time
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline


# 1. Data-prep
dataset = load_wine()
x = dataset.data   
y = dataset.target 

kfold = KFold(n_splits=5)

parameters = [
    {'randomforestclassifier__n_jobs': [-1], 'randomforestclassifier__n_estimators': [100, 200], 'randomforestclassifier__max_depth': [6, 8, 10], 'randomforestclassifier__min_samples_leaf': [5, 7, 10]},
    {'randomforestclassifier__n_jobs': [-1], 'randomforestclassifier__max_depth': [5, 6, 7, 9], 'randomforestclassifier__min_samples_leaf': [3, 6, 9, 11],  'randomforestclassifier__min_samples_split': [3, 4, 5]},
    {'randomforestclassifier__n_jobs': [-1], 'randomforestclassifier__min_samples_leaf': [3, 5, 7],  'randomforestclassifier__min_samples_split': [2, 3, 5, 10]},
    {'randomforestclassifier__n_jobs': [-1], 'randomforestclassifier__min_samples_split': [2, 3, 5, 10]},
]


# 2. Modelling
pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
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
GridSearch took 5.373067617416382 seconds
best esimator was Pipeline(MinMaxScaler(), RandomForestClassifier(min_samples_split=5, n_jobs=-1))])
best parameter was {'randomforestclassifier__n_jobs': -1, 'randomforestclassifier__min_samples_split': 5}
score: 1.0
'''