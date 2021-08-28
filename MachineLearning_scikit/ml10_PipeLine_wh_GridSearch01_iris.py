import time
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline


# 1. Data-prep
dataset = load_iris()
x = dataset.data   
y = dataset.target 

kfold = KFold(n_splits=5)

# below is not-subscribable command, parameters should be applied on RF in pipe
# parameters = [
#     {'n_jobs': [-1], 'n_estimators': [100, 200], 'max_depth': [6, 8, 10], 'min_samples_leaf': [5, 7, 10]},
#     {'n_jobs': [-1], 'max_depth': [5, 6, 7, 9], 'min_samples_leaf': [3, 6, 9, 11],  'min_samples_split': [3, 4, 5]},
#     {'n_jobs': [-1], 'min_samples_leaf': [3, 5, 7],  'min_samples_split': [2, 3, 5, 10]},
#     {'n_jobs': [-1], 'min_samples_split': [2, 3, 5, 10]},
# ]

# NameOFModel__NameOfParameter: [factors...]
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
GridSearch took 5.35440731048584 seconds
best esimator was Pipeline(steps=[MinMaxScaler(), RandomForestClassifier(max_depth=6, min_samples_leaf=3, min_samples_split=3, n_jobs=-1))])
best parameter was {'randomforestclassifier__n_jobs': -1, 'randomforestclassifier__min_samples_split': 3, 'randomforestclassifier__min_samples_leaf': 3, 'randomforestclassifier__max_depth': 6}
score: 0.9733333333333334
'''