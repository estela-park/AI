import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor


# Data-prep
x, y = load_diabetes(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=96)


# pre-Fitting the estimator 
model = XGBRegressor(n_jobs=-1)
# n_jobs (Optional[int]) – Number of parallel threads used to run xgboost.
model.fit(x_train, y_train)
print('pre-feature engineering:', model.score(x_train, y_train))
thresholds = np.sort(model.feature_importances_) # ndarray

for bar in thresholds:
    selector = SelectFromModel(model, threshold=bar, prefit=True)
    x_train_engineered = selector.transform(x_train)
    x_test_engineered = selector.transform(x_test)

    temp = XGBRegressor(n_jobs=-1)
    temp.fit(x_train_engineered, y_train)
    print('Threshold: %.2f, score: %.4f' %(bar, temp.score(x_test_engineered, y_test))) 

'''
**BOSTON**
pre-feature engineering: 0.9999
Threshold: 0.00,  score: 0.9083
Threshold: 0.01,  score: 0.9051
Threshold: 0.01,  score: 0.9104
Threshold: 0.01,  score: 0.9036
Threshold: 0.01,  score: 0.9017
Threshold: 0.01,  score: 0.9019
Threshold: 0.01,  score: 0.9100
Threshold: 0.02,  score: 0.9125
Threshold: 0.05,  score: 0.9053
Threshold: 0.06,  score: 0.8764
Threshold: 0.07,  score: 0.8428
Threshold: 0.11,  score: 0.7155
Threshold: 0.63,  score: 0.4495

**DIABETES**
pre-feature engineering: 0.9999
Threshold: 0.03,  score: 0.4180
Threshold: 0.03,  score: 0.3744
Threshold: 0.05, score: 0.3993
Threshold: 0.05, score: 0.4175
Threshold: 0.05, score: 0.3984
Threshold: 0.06, score: 0.3552
Threshold: 0.08, score: 0.2873
Threshold: 0.16, score: 0.3140
Threshold: 0.21, score: 0.1805
Threshold: 0.28, score: 0.0882
'''