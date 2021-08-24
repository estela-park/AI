from json import load
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


# 1. Data-prep
x, y = load_boston(return_X_y=True)
# (506, 13) (506,)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=78)
# (379, 13) (127, 13) (379,) (127,)


# 2. Modelling
model = XGBRegressor(n_jobs=-1)
# n_jobs (Optional[int]) – Number of parallel threads used to run xgboost.


# 3. Training
