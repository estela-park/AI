import joblib
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score


# 1. Data-prep
datasets = load_boston()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=78)

scaler_mm = MinMaxScaler()
x_train_mm = scaler_mm.fit_transform(x_train)
x_test_mm = scaler_mm.transform(x_test)

# 2. Model
model = joblib.load('../_save/_XGB/m22_joblib.dat')


score = model.score(x_test_mm, y_test)
predict = model.predict(x_test_mm)
r2 = r2_score(y_test, predict)
print('model.score:', score, '& model.R2score:', r2)

# where it is stored, model.score: 0.8159750970042733 & model.R2score: 0.8159750970042733
# where is is loaded, model.score: 0.8159750970042733 & model.R2score: 0.8159750970042733

