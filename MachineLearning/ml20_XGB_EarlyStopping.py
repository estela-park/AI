import time
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


# 1. Data-prep
datasets = load_boston()
x = datasets['data']
y = datasets['target']
# (506, 13) (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=78)

scaler_mm = MinMaxScaler()
x_train_mm = scaler_mm.fit_transform(x_train)
x_test_mm = scaler_mm.transform(x_test)

# 2. Modelling
model = XGBRegressor(n_estimators=2400, learning_rate=0.01, n_jobs=1)
# n_jobs: the number of cores to utiltize


# Training & Evaluation
start = time.time()
model.fit(x_train_mm, y_train, verbose=1, eval_metric='rmse', eval_set=[(x_test_mm, y_test)], early_stopping_rounds=4)
# early_stopping_rounds corresponds to patience
end = time.time() - start

score = model.score(x_test_mm, y_test)
predict = model.predict(x_test_mm)
r2 = r2_score(y_test, predict)
print('*****************With MinMaxScaler*****************')
print('it took', end//60, 'minutes and', end%60,'seconds')
print('model.score:', score, '& model.R2score:', r2)

'''
+++++++++epochs=2400, early_stopping_rounds=4, stopped @618+++++++++
[0]     validation_0-rmse:23.05024
...
[618]   validation_0-rmse:2.19458
*****************With MinMaxScaler*****************
it took 0.0 minutes and 0.8225910663604736 seconds
model.score: 0.9252966909900735 & model.R2score: 0.9252966909900735
'''