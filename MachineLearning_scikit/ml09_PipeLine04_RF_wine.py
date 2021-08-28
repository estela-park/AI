import numpy as np
import time
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline


# 1. Data-prep
dataset = load_wine()
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
time spent: 0.09 seconds
model.score: 1.0
'''