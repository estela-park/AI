import time
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


# 1. Data-prep
datasets = load_diabetes()
x = datasets['data']
y = datasets['target']
# (506, 13) (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=78)

scaler_mm = MinMaxScaler()
x_train_mm = scaler_mm.fit_transform(x_train)
x_test_mm = scaler_mm.transform(x_test)

scaler_std = StandardScaler()
x_train_std = scaler_std.fit_transform(x_train)
x_test_std = scaler_std.transform(x_test)


# 2. Modelling
model = XGBRegressor(n_estimators=240, learning_rate=0.03, n_jobs=1)


# 3-1. Training & Evaluation: Minmax
start = time.time()
model.fit(x_train_mm, y_train, verbose=1, eval_metric='rmse', eval_set=[(x_test_mm, y_test)])
end = time.time() - start

score = model.score(x_test_mm, y_test)
predict = model.predict(x_test_mm)
r2 = r2_score(y_test, predict)
print('*****************With MinMaxScaler*****************')
print('it took', end//60, 'minutes and', end%60,'seconds')
print('model.score:', score, '& model.R2score:', r2)

print('==========================EVAL==========================')
print('eval.result:', model.evals_result())


# 3-2. Training & Evaluation: Standard
start = time.time()
model.fit(x_train_std, y_train, verbose=1, eval_metric='rmse', eval_set=[(x_test_std, y_test)])
end = time.time() - start

score = model.score(x_test_std, y_test)
predict = model.predict(x_test_std)
r2 = r2_score(y_test, predict)
print('*****************With StandardScaler*****************')
print('it took', end//60, 'minutes and', end%60,'seconds')
print('model.score:', score, '& model.R2score:', r2)

print('==========================EVAL==========================')
print('eval.result:', model.evals_result())


'''
[0]     validation_0-rmse:168.58931
...
[239]   validation_0-rmse:61.67027
*****************With MinMaxScaler*****************
it took 0.0 minutes and 0.3281209468841553 seconds
model.score: 0.37212595262659753 & model.R2score: 0.37212595262659753
==========================EVAL==========================
eval.result: {'validation_0': OrderedDict([('rmse', [168.58931, ... 61.670269])])}
[0]     validation_0-rmse:168.58931
...
[239]   validation_0-rmse:61.48742
*****************With StandardScaler*****************
it took 0.0 minutes and 0.26880335807800293 seconds
model.score: 0.37584362419142936 & model.R2score: 0.37584362419142936
==========================EVAL==========================
eval.result: {'validation_0': OrderedDict([('rmse', [168.58931, ... 61.487419])])}
PS D:\study> 
'''