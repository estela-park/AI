import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline


# 1. Data-prep
dataset = load_breast_cancer()
x = dataset.data   
y = dataset.target 


# 2. Modelling
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())


# 3. Training
start = time.time()
model.fit(x, y)
end = time.time() - start


# 4. Evaluation
print('time spent:', end, 'seconds')
print('model.score:', model.score(x, y))

'''
time spent: 0.14 seconds
model.score: 1.0
'''