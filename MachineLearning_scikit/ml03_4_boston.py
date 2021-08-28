import numpy as np
import time
from sklearn.datasets import load_boston
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
# Since regressor gives continuous value of function and classifier discrete,
# boston's label, value of house, should be calculated with regressor
from sklearn.neighbors import KNeighborsRegressor
# eventhough LogisticRegression has Regression in its name, it does not.
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# performance index for continuous label: R^2, discrete: accuracy
from sklearn.metrics import r2_score

dataset = load_boston()

x = dataset.data   
# (506, 13)
y = dataset.target
# (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85)

max_abs_scaler = MaxAbsScaler()
max_abs_scaler.fit(x_train)
x_train_ma = max_abs_scaler.transform(x_train)
x_test_ma = max_abs_scaler.transform(x_test)

start = time.time()

model_ma = RandomForestRegressor()

# Linear SVC doesn't do regression

# except data, most are set default
model_ma.fit(x_train_ma, y_train)

# evaluate => score, it gives back accuracy.
result_ma = model_ma.score(x_test_ma, y_test)
predict_ma = model_ma.predict(x_test_ma)

end = time.time() - start

# Accuracy computing
acc = r2_score(y_test, predict_ma)

print('it took',end)
print('accuracy score:', acc, end='')
print(', prediction for ',y_test[:8],'is',predict_ma[:8])

'''
**KNeighborsRegressor
    accuracy score: 0.6782135294911238
    prediction for  [17.6  7.5 36.2 25.  26.7 19.8 27.9 17.2] is [19.18 10.34 30.78 24.72 25.26 20.12 31.52 11.72]
**LinearRegression
    accuracy score: 0.7532541438495368
    prediction for  [20.4 22.9 24.8 17.1 14.1 20.4 26.5 20.3] is [22.86 20.54 26.84 19.93 16.27 20.28 25.41 22.23]
**DecisionTreeRegressor
    accuracy score: 0.8295619988980103
    prediction for  [30.1 22.9 36.4 16.  20.4 22.7 18.7 22.8] is [31.6 22.6 34.9 23.1 24.1 19.1 19.  30.8]
**RandomForestRegressor
    accuracy score: 0.8254297676525074
    prediction for  [21.  15.6 31.5 32.  19.9 23.1 15.2 28.7] is [21.08 15.85 29.18 27.74 20.55 23.91 15.36 24.26]
'''