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

standard_scaler = StandardScaler()
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
  for MaxAbsScaler,
   - 
  for RobustScaler,
   - 
  for QuantileTransformer,
   - 
  for PowerTransformer,
   - 
'''