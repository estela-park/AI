from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


# 1. Data-prep
datasets = load_boston()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=78)

scaler_mm = MinMaxScaler()
x_train_mm = scaler_mm.fit_transform(x_train)
x_test_mm = scaler_mm.transform(x_test)

# 2. Model
# model precursor should be created first then weight is loaded separately
model = XGBRegressor()
model.load_model('../_save/_XGB/ml23_save_model.dat')


score = model.score(x_test_mm, y_test)
predict = model.predict(x_test_mm)
r2 = r2_score(y_test, predict)
print('model.score:', score, '& model.R2score:', r2)