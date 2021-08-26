from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

datasets = load_diabetes()

x = datasets.data   
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=72)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train)
x_train_mm = min_max_scaler.transform(x_train)
x_test_mm = min_max_scaler.transform(x_test)

standard_scaler = StandardScaler(with_std=False)
standard_scaler.fit(x_train)
x_train_st = standard_scaler.transform(x_train)
x_test_st = standard_scaler.transform(x_test)

start = time.time()

model_mm = Sequential()
model_mm.add(Dense(64, input_shape=(10,), activation='relu'))
model_mm.add(Dense(128, activation='relu'))
model_mm.add(Dense(32, activation='relu'))
model_mm.add(Dense(1))

model_mm.compile(loss='mse', optimizer='adam', metrics=['mae'])
model_mm.fit(x_train_mm, y_train, epochs=240, validation_split=0.15, verbose=0, batch_size=32)

model_st = Sequential()
model_st.add(Dense(64, input_shape=(10,), activation='relu'))
model_st.add(Dense(128, activation='relu'))
model_st.add(Dense(32, activation='relu'))
model_st.add(Dense(1))

model_st.compile(loss='mse', optimizer='adam', metrics=['mae'])
model_st.fit(x_train_st, y_train, epochs=240, validation_split=0.15, verbose=0, batch_size=32)

loss_mm = model_mm.evaluate(x_test_mm, y_test)
predict_mm = model_mm.predict(x_test_mm)

loss_st = model_st.evaluate(x_test_st, y_test)
predict_st = model_st.predict(x_test_st)

end = time.time() - start

r2_mm = r2_score(y_test, predict_mm)
r2_st = r2_score(y_test, predict_st)

print('for minmax-scaler, loss:', loss_mm, 'accuracy:', r2_mm)
print('for standard-scaler, loss:', loss_st, 'accuracy:', r2_st)

'''
Before scaling,
  - loss: [2109.333251953125, 37.41487503051758], accuracy: 0.6805598196277636
After scaling, hyperparameters are set identically
  for minmax-scaler, 
   - loss: [2264.997314453125, 39.101959228515625] accuracy: 0.6630762302852726
  for standard-scaler,
   - loss: [3097.4091796875, 44.518802642822266] accuracy: 0.5309243503957015
  for standard-scaler, with_mean=False
   - loss: [3459.64404296875, 47.10385513305664] accuracy: 0.47606700440569405
   for standard-scaler, with_std=False
   - loss: [2076.2578125, 36.727149963378906] accuracy: 0.6855688057664485
** minmax doesn't make much difference, standard-scaler made the accuracy worse.
   guessing the raw data is evenly distributed already.
** how standardizing works x_scaled = (x - u) / s
   - with_mean: True(default); u is the mean of the training samples/ False; u=0
   - with_std : True(default); s is the standard deviation of the training samples/ False; s=1
** standard deviations for x_train
    0.02756302058522011
    0.034007999551426665
    0.026335128018073504
    0.03795319153766971
    0.027275718491703287
    0.03601335984969945
    0.05179851408579541
    0.04514590871987909
    0.03900843634677638
    0.04183835679285838
    0.02605552033832037
    0.0382091301861343
    0.03213239002038584
** implication: 데이터가 0 에서 많이 벗어나 있어서 -u 는 필요하고, 기존의 표준편차가 많이 작기 때문에 /s 를 하면 데이터가 너무 퍼지게 된다.
'''