import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# 1. Data-prep
datasets = load_breast_cancer()
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
model = XGBClassifier(n_estimators=240, learning_rate=0.03, n_jobs=1)


# 3-1. Training & Evaluation: Minmax
start = time.time()
model.fit(x_train_mm, y_train, verbose=1, eval_metric='logloss', eval_set=[(x_test_mm, y_test)])
end = time.time() - start

score = model.score(x_test_mm, y_test)
predict = model.predict(x_test_mm)
acc = accuracy_score(y_test, predict)
print('*****************With MinMaxScaler*****************')
print('it took', end//60, 'minutes and', end%60,'seconds')
print('model.score:', score, '& model.acc:', acc)

print('==========================EVAL==========================')
print('eval.result:', model.evals_result())


# 3-2. Training & Evaluation: Standard
start = time.time()
model.fit(x_train_std, y_train, verbose=1, eval_metric='logloss', eval_set=[(x_test_std, y_test)])
end = time.time() - start

score = model.score(x_test_std, y_test)
predict = model.predict(x_test_std)
acc = accuracy_score(y_test, predict)
print('*****************With StandardScaler*****************')
print('it took', end//60, 'minutes and', end%60,'seconds')
print('model.score:', score, '& model.acc:', acc)

print('==========================EVAL==========================')
print('eval.result:', model.evals_result())


'''
[0]     validation_0-logloss:0.67192
...
[239]   validation_0-logloss:0.20329
*****************With MinMaxScaler*****************
it took 0.0 minutes and 0.45543718338012695 seconds
model.score: 0.9186046511627907 & model.acc: 0.9186046511627907
==========================EVAL==========================
eval.result: {'validation_0': OrderedDict([('logloss', [0.671917, ... 0.203286])])}


[0]     validation_0-logloss:0.67254
...
[239]   validation_0-logloss:0.21020
*****************With StandardScaler*****************
it took 0.0 minutes and 0.45931506156921387 seconds
model.score: 0.9069767441860465 & model.acc: 0.9069767441860465
==========================EVAL==========================
eval.result: {'validation_0': OrderedDict([('logloss', [0.672545, ... 0.210197])])}
'''