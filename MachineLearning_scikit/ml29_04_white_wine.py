import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


data = pd.read_csv('../_data/white_wine.csv', sep=';', index_col=None, header=0)

data = data.values
x = data[:, :11]
y = data[:, 11]

newlist = []
for i in list(y):
    if i <= 4:
        newlist += [0]
    elif i <= 7:
        newlist += [1]
    else:
        newlist += [2]

y = pd.DataFrame(np.array(newlist))
count_label = y.groupby(0)[0].count()

'''
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5

label editted
0
0     183
1    4535
2     180
'''

x_train, x_test, y_train, y_test = train_test_split(data[:, :11], y, random_state=72)

scaler_mm = MinMaxScaler()
scaler_std = StandardScaler()
scaler_rb = RobustScaler()

x_train_mm = scaler_mm.fit_transform(x_train)
x_train_std = scaler_std.fit_transform(x_train)
x_train_rb = scaler_rb.fit_transform(x_train)
x_test_mm = scaler_mm.transform(x_test)
x_test_std = scaler_std.transform(x_test)
x_test_rb = scaler_rb.transform(x_test)

model = XGBClassifier(n_jobs=-1)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)

model.fit(x_train_mm, y_train)
score_mm = model.score(x_test_mm, y_test)

model.fit(x_train_std, y_train)
score_std = model.score(x_test_std, y_test)

model.fit(x_train_rb, y_train)
score_rb = model.score(x_test_rb, y_test)

print('without any scaler, accuracy:', score)
print('with MinMaxScaler, accuracy:', score_mm)
print('with StandardScaler, accuracy:', score_std)
print('with RobustScaler, accuracy:', score_rb)


'''
- Previous Result
it took 2 minutes and 25 seconds
for robust: [3.2979772090911865, 0.6448979377746582]
for powerT: [3.2596006393432617, 0.6503401398658752]

-XG Booster without label editting
without any scaler, accuracy: 0.6636734693877551
with MinMaxScaler, accuracy: 0.6628571428571428
with StandardScaler, accuracy: 0.6644897959183673
with RobustScaler, accuracy: 0.6628571428571428

-XG Booster with label editting
without any scaler, accuracy: 0.9371428571428572
with MinMaxScaler, accuracy: 0.9371428571428572
with StandardScaler, accuracy: 0.9371428571428572
with RobustScaler, accuracy: 0.9371428571428572

It was imbalance that's the reason for low performance
'''