import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor


# 1: Data-prep
dataset = load_boston()

x = dataset.data   
y = dataset.target 

kfold = KFold(n_splits=5)


# 2: Modelling
model = RandomForestRegressor()


# 3: Training
#            > not needed


# 4: Evaluation
scores = cross_val_score(model, x, y, cv=kfold)
print('model: RandomForestRegressor, k-value: 5, accuracy:', scores, round(np.mean(scores), 4))

'''
model: KNeighborsRegressor,   k-value: 5, accuracy: [-1.10921186  0.14934963 -0.4259195  -0.01474393 -0.17455668] -0.315
model: LinearRegression,      k-value: 5, accuracy: [ 0.63919994  0.71386698  0.58702344  0.07923081 -0.25294154] 0.3533
model: DecisionTreeRegressor, k-value: 5, accuracy: [ 0.67255059  0.43086962  0.62215815  0.39192204 -1.37538635] 0.1484
model: RandomForestRegressor, k-value: 5, accuracy: [0.77050911 0.84805521 0.70742304 0.46655406 0.33868462] 0.6262
'''