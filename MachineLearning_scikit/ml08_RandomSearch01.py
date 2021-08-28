'''
GridSearch looks for the optimal hyper-parameter nondiscriminately, 
which means it wastes significant amound of time dealing with useless parameters.
Like a Dropout layer in DNN models, RandomSearch drops hyper-parameters in random fashion
'''
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
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
model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1)
model_r = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=1)


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
Fitting 5 folds for each of 18 candidates, totalling 90 fits
Fitting 5 folds for each of 10 candidates, totalling 50 fits
GridSearch took 0.13663411140441895 seconds
RandomSearch took 0.06582379341125488 seconds
best esimator was SVC(C=1, kernel='linear')
best parameter was {'C': 1, 'kernel': 'linear'}
Random_best estimator was SVC(C=1000, gamma=0.001, kernel='sigmoid')
Random_best parameter was {'kernel': 'sigmoid', 'gamma': 0.001, 'C': 1000}
score: 0.9933333333333333
Random_score: 0.9933333333333333
'''