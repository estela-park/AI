# To print progress status of training
# as model.fit arguments, give
# eval_set=[(x, y), ...]
# eval_metrics='any object function'
#              > default: rmse
# verbose=#

import time
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

scaler_std = StandardScaler()
x_train_std = scaler_std.fit_transform(x_train)
x_test_std = scaler_std.transform(x_test)


# 2. Modelling
model = XGBRegressor(n_estimators=240, learning_rate=0.01, n_jobs=1)
# n_jobs: the number of cores to utiltize


# 3-1. Training & Evaluation: Minmax
start = time.time()
model.fit(x_train_mm, y_train, verbose=1, eval_metric=['rmse', 'logloss'], eval_set=[(x_train_std, y_train), (x_test_mm, y_test)])
end = time.time() - start

score = model.score(x_test_mm, y_test)
predict = model.predict(x_test_mm)
r2 = r2_score(y_test, predict)
print('*****************With MinMaxScaler*****************')
print('it took', end//60, 'minutes and', end%60,'seconds')
print('model.score:', score, '& model.R2score:', r2)

print('==========================EVAL==========================')
aaa = model.evals_result()
print('eval.result:', model.evals_result())


# 3-2. Training & Evaluation: Standard
start = time.time()
model.fit(x_train_std, y_train, verbose=1, eval_metric=['rmse', 'logloss'], eval_set=[(x_train_std, y_train), (x_test_std, y_test)])
# eval_set are printed as validation_#
end = time.time() - start

score = model.score(x_test_std, y_test)
predict = model.predict(x_test_std)
r2 = r2_score(y_test, predict)
print('*****************With StandardScaler*****************')
print('it took', end//60, 'minutes and', end%60,'seconds')
print('model.score:', score, '& model.R2score:', r2)

print('==========================EVAL==========================')
bbb = model.evals_result()
print('eval.result:', model.evals_result())


'''
[0]     validation_0-rmse:23.05024
...
[239]   validation_0-rmse:3.44306
*****************With MinMaxScaler*****************
it took 0.0 minutes and 0.34482598304748535 seconds
model.score: 0.8159750970042733 & model.R2score: 0.8159750970042733
==========================EVAL==========================
eval.result: {'validation_0': OrderedDict([('rmse', [23.050241, ... 3.443058])])}


[0]     validation_0-rmse:23.05024
...
[239]   validation_0-rmse:3.44293
*****************With StandardScaler*****************
it took 0.0 minutes and 0.2838864326477051 seconds
model.score: 0.8159883978227769 & model.R2score: 0.8159883978227769
==========================EVAL==========================
eval.result: {'validation_0': OrderedDict([('rmse', [23.050241, ... 3.442933])])}
'''

epochs = len(aaa['validation_0']['rmse'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, aaa['validation_0']['rmse'], label='Train')
ax.plot(x_axis, aaa['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('RMSE')
plt.title('XGBoost RMSE_ MinMax scaled')

fig, ax = plt.subplots()
ax.plot(x_axis, bbb['validation_0']['logloss'], label='Train')
ax.plot(x_axis, bbb['validation_1']['logloss'], label='Test')
ax.legend()

plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss_ Standard scaled')
plt.show()

'''
eval_metric's choices:

-rmse: root mean square error
-rmsle: root mean square log error
-mae: mean absolute error
-mape: mean absolute percentage error
-mphe: mean Pseudo Huber error. Default metric of reg:pseudohubererror objective.
-logloss: negative log-likelihood
-error: Binary classification error rate. It is calculated as #(wrong cases)/#(all cases).
-error@t: a different than 0.5 binary classification threshold value could be specified 
          by providing a numerical value through ‘t’.
-merror: Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases).
-mlogloss: Multiclass logloss.
-auc: Receiver Operating Characteristic Area under the Curve. 
-aucpr: Area under the PR curve. Available for binary classification and learning-to-rank tasks.
-ndcg: Normalized Discounted Cumulative Gain
-map: Mean Average Precision
-ndcg@n, map@n: ‘n’ can be assigned as an integer to cut off the top positions in the lists for evaluation.
-poisson-nloglik: negative log-likelihood for Poisson regression
-gamma-nloglik: negative log-likelihood for gamma regression
-cox-nloglik: negative partial log-likelihood for Cox proportional hazards regression
-gamma-deviance: residual deviance for gamma regression
-tweedie-nloglik: negative log-likelihood for Tweedie regression 
                  (at a specified value of the tweedie_variance_power parameter)
-aft-nloglik: Negative log likelihood of Accelerated Failure Time model.
-interval-regression-accuracy: Fraction of data points whose predicted labels 
                               fall in the interval-censored labels. 
                               Only applicable for interval-censored data.
'''