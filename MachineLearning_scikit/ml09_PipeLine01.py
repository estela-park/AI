# Scaling + ModelSelection + CrossValidation + Hyper-ParameterTuning (+randomised dropping of hyper-parmeters)

import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline


# 1. Data-prep
dataset = load_iris()
x = dataset.data   
y = dataset.target 


# 2. Modelling
model = make_pipeline(MinMaxScaler(), SVC())


# 3. Training
start = time.time()
model.fit(x, y)
end = time.time() - start


# 4. Evaluation
print('time spent:', end, 'seconds')
print('model.score:', model.score(x, y))

'''
time spent: 0.005 seconds
model.score: 0.98
'''