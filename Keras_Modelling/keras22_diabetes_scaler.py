import time
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

datasets = load_diabetes()

x = datasets.data   
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=72)

max_abs_scaler = MaxAbsScaler()
max_abs_scaler.fit(x_train)
x_train_ma = max_abs_scaler.transform(x_train)
x_test_ma = max_abs_scaler.transform(x_test)

robust_scaler = RobustScaler()
robust_scaler.fit(x_train)
x_train_rb = robust_scaler.transform(x_train)
x_test_rb = robust_scaler.transform(x_test)

power_scaler = PowerTransformer()
power_scaler.fit(x_train)
x_train_pt = power_scaler.transform(x_train)
x_test_pt = power_scaler.transform(x_test)

quantile_scaler = QuantileTransformer()
quantile_scaler.fit(x_train)
x_train_qt = quantile_scaler.transform(x_train)
x_test_qt = quantile_scaler.transform(x_test)

start = time.time()

model_ma = Sequential()
model_ma.add(Dense(64, input_shape=(10,), activation='relu'))
model_ma.add(Dense(128, activation='relu'))
model_ma.add(Dense(32, activation='relu'))
model_ma.add(Dense(1))

model_ma.compile(loss='mse', optimizer='adam', metrics=['mae'])
model_ma.fit(x_train_ma, y_train, epochs=240, validation_split=0.15, verbose=0, batch_size=32)

model_rb = Sequential()
model_rb.add(Dense(64, input_shape=(10,), activation='relu'))
model_rb.add(Dense(128, activation='relu'))
model_rb.add(Dense(32, activation='relu'))
model_rb.add(Dense(1))

model_rb.compile(loss='mse', optimizer='adam', metrics=['mae'])
model_rb.fit(x_train_rb, y_train, epochs=240, validation_split=0.15, verbose=0, batch_size=32)

model_pt = Sequential()
model_pt.add(Dense(64, input_shape=(10,), activation='relu'))
model_pt.add(Dense(128, activation='relu'))
model_pt.add(Dense(32, activation='relu'))
model_pt.add(Dense(1))

model_pt.compile(loss='mse', optimizer='adam', metrics=['mae'])
model_pt.fit(x_train_pt, y_train, epochs=240, validation_split=0.15, verbose=0, batch_size=32)

model_qt = Sequential()
model_qt.add(Dense(64, input_shape=(10,), activation='relu'))
model_qt.add(Dense(128, activation='relu'))
model_qt.add(Dense(32, activation='relu'))
model_qt.add(Dense(1))

model_qt.compile(loss='mse', optimizer='adam', metrics=['mae'])
model_qt.fit(x_train_qt, y_train, epochs=240, validation_split=0.15, verbose=0, batch_size=32)

loss_ma = model_ma.evaluate(x_test_ma, y_test)
predict_ma = model_ma.predict(x_test_ma)

loss_rb = model_rb.evaluate(x_test_rb, y_test)
predict_rb = model_rb.predict(x_test_rb)

loss_pt = model_pt.evaluate(x_test_pt, y_test)
predict_pt = model_pt.predict(x_test_pt)

loss_qt = model_qt.evaluate(x_test_qt, y_test)
predict_qt = model_qt.predict(x_test_qt)

end = time.time() - start

r2_ma = r2_score(y_test, predict_ma)
r2_rb = r2_score(y_test, predict_rb)
r2_pt = r2_score(y_test, predict_pt)
r2_qt = r2_score(y_test, predict_qt)

print('for maxabsolute scaler, loss:', loss_ma, 'accuracy:', r2_ma)
print('for robust scaler, loss:', loss_rb, 'accuracy:', r2_rb)
print('for power transformer, loss:', loss_pt, 'accuracy:', r2_pt)
print('for quantile transformer, loss:', loss_qt, 'accuracy:', r2_qt)

'''
Before scaling,
  - loss: [2109.333251953125, 37.41487503051758], accuracy: 0.6805598196277636
After scaling, hyperparameters are set identically
  for MaxAbsScaler,
   - loss: [2487.573486328125, 39.96940231323242] accuracy: 0.6232786467667979
   - loss: [2451.08154296875, 39.009193420410156] accuracy: 0.6288050498927467
   - loss: [2483.04052734375, 39.04961395263672] accuracy: 0.6239650918751773
  for RobustScaler,
   - loss: [3236.02490234375, 46.81381607055664] accuracy: 0.5099321594178154
   - loss: [3623.12109375, 49.16351318359375] accuracy: 0.4513098384265535
   - loss: [3325.814208984375, 46.9292106628418] accuracy: 0.496334370952738
  for PowerTransformer,
   - loss: [3308.165771484375, 46.676570892333984] accuracy: 0.49900700917028784
   - loss: [4058.00537109375, 52.4571418762207] accuracy: 0.38545041050337936
   - loss: [3299.63916015625, 46.630332946777344] accuracy: 0.5002983984423002
  for QuantileTransformer,
   - loss: [2307.609375, 39.26508331298828] accuracy: 0.6505326329820283
   - loss: [2373.011962890625, 40.24491882324219] accuracy: 0.6406279745592962
   - loss: [2407.8251953125, 40.083412170410156] accuracy: 0.6353558343280721
** implication: 
   - parameters are tuned according to un-scaled data, for that MaxabsScaler and QuantileTransformer work relatively well
     for this data-set minmax and maxabs both works fine which means that data-set doesn't have large number of outliers.
     other possibility is, that since the raw data has narrow range, both scalers futhers the data away making it more discrete.
   - un-scaled data's range: very narrow
      for 0 column: 0.0506801187398187 -0.0400993174922969
      for 1 column: 0.0817644407962278 -0.00914709342983014
      for 2 column: 0.0506801187398187 -0.0359677812752396
      for 3 column: 0.0859065477110625 -0.0342290680567117
      for 4 column: 0.00114379737951254 -0.081413765817132
      for 5 column: 0.0569117993072195 -0.0563700932930843
      for 6 column: 0.12732761685941 -0.0581273968683752
      for 7 column: 0.092004364187062 -0.044641636506989
      for 8 column: 0.0506801187398187 -0.0891974838246376
      for 9 column: 0.0633666506664982 -0.0545491159304391
      for 10 column: 0.0118237214092792 -0.0607565416547144
      for 11 column: 0.0449584616460628 -0.0814798836443389
      for 12 column: 0.063186803319791 -0.0353068801305926
   - I estimate that the data would be uniformly distributed(not Gaussian) with stable variance and minimal skewness.
     Robust will eliminate outer quantile not removing outliers which is few.
     Power Transformer map the data to Gaussian distribution which is not very compatible with the data.
     Quantile Transformer map the data to Uniform distribution which is compatible with the data.
'''