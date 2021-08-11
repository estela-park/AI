import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# 1: Data-prep
dataset = load_diabetes()

x = dataset.data   
y = dataset.target 

kfold = KFold(n_splits=5)


# 2: Modelling
model = LinearRegression()


# 3: Training
#            > not needed


# 4: Evaluation
scores = cross_val_score(model, x, y, cv=kfold)
print('model: LinearRegression, k-value: 5, accuracy:', scores, round(np.mean(scores), 4))

'''
model: KNeighborsRegressor,   k-value: 5, accuracy: [-0.28054359 -0.06099382  0.00515191  0.01225025 -0.30457613] -0.1257
model: LinearRegression,      k-value: 5, accuracy: [0.42955643 0.52259828 0.4826784  0.42650827 0.55024923] 0.4823
model: DecisionTreeRegressor, k-value: 5, accuracy: [-0.34232993 -0.02561467 -0.05868949  0.10838135 -0.17539964] -0.0987
model: RandomForestRegressor, k-value: 5, accuracy: [0.38852998 0.50225806 0.42695839 0.35795961 0.4339548 ] 0.4219
'''