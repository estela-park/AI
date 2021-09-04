import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

data = pd.read_csv('../_data/white_wine.csv', sep=';', index_col=None, header=0)

print(data.shape)
# (4898, 12)
print(data.info())
print(data.describe())
print(data.head(5))

# Turning DataFrame into ndarray
data = data.values

x_train, x_test, y_train, y_test = train_test_split(data[:, :11], data[:, 11], random_state=72)

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

-XG Booster
without any scaler, accuracy: 0.6636734693877551
with MinMaxScaler, accuracy: 0.6628571428571428
with StandardScaler, accuracy: 0.6644897959183673
with RobustScaler, accuracy: 0.6628571428571428
'''