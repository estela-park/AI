# Does SelectFromModel work along with GridSearch?

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor

parameters = [{
    'n_estimators': [30, 50], 'learning_rate': [0.01, 0.001], 'base_score': [0.1, 0.2]
}]

# Data-prep
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=96)


# pre-Fitting the estimator 
# model = GridSearchCV(XGBRegressor(), parameters, verbose=2)
model = XGBRegressor(base_score=0.2, learning_rate=0.01, n_estimators=50, n_jobs=-1)
model.fit(x_train, y_train)
print('pre-feature engineering:', model.score(x_train, y_train))
thresholds = np.sort(model.feature_importances_) # ndarray

for bar in thresholds:
    selector = SelectFromModel(model, threshold=bar, prefit=True)
    x_train_engineered = selector.transform(x_train)
    x_test_engineered = selector.transform(x_test)

    temp = XGBRegressor(base_score=0.2, learning_rate=0.01, n_estimators=50, n_jobs=-1)
    temp.fit(x_train_engineered, y_train)
    print('Threshold: %.2f, score: %.4f' %(bar, temp.score(x_test_engineered, y_test))) 


'''
**Manual Model**
pre-feature engineering: 0.9999
Threshold: 0.03,  score: 0.4180
Threshold: 0.03,  score: 0.3744

**GridSearch**
pre-feature engineering: -1.0242
Threshold: 0.03,  score: -1.3154
Threshold: 0.03,  score: -1.3066
Threshold: 0.03,  score: -1.3053
Threshold: 0.03,  score: -1.2990
Threshold: 0.03,  score: -1.2820
Threshold: 0.04,  score: -1.2490
Threshold: 0.05,  score: -1.2546
Threshold: 0.06,  score: -1.2499
Threshold: 0.30,  score: -1.2339
Threshold: 0.40,  score: -1.2543
'''