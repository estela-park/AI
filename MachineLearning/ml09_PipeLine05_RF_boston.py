import numpy as np
import time
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline


# 1. Data-prep
dataset = load_boston()
x = dataset.data   
y = dataset.target 


# 2. Modelling
model = make_pipeline(MinMaxScaler(), RandomForestRegressor())


# 3. Training
start = time.time()
model.fit(x, y)
end = time.time() - start


# 4. Evaluation
print('time spent:', end, 'seconds')
print('model.score:', model.score(x, y))

'''
time spent: 0.23 seconds
model.score: 0.9821139549772063
'''