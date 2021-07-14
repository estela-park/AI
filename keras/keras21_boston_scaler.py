from sklearn.datasets import load_boston
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MaxAbsScaler, PowerTransformer, QuantileTransformer, RobustScaler
import numpy as np

dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=82)

for i in range(13):
  print(np.std(x_train[i]))

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

input1 = Input(shape=(13, ))
dense1 = Dense(104, activation='relu')(input1)
dense2 = Dense(52, activation='relu')(dense1)
dense3 = Dense(26, activation='relu')(dense2)
dense4 = Dense(13, activation='relu')(dense3)
output1 = Dense(1)(dense4)

model_ma = Model(inputs=input1, outputs=output1)

model_ma.compile(loss='mse', optimizer='adam')
model_ma.fit(x_train_ma, y_train, batch_size=13, epochs=300, verbose=0, validation_split=0.15)

model_rb = Model(inputs=input1, outputs=output1)

model_rb.compile(loss='mse', optimizer='adam')
model_rb.fit(x_train_rb, y_train, batch_size=13, epochs=300, verbose=0, validation_split=0.15)

model_pt = Model(inputs=input1, outputs=output1)

model_pt.compile(loss='mse', optimizer='adam')
model_pt.fit(x_train_pt, y_train, batch_size=13, epochs=300, verbose=0, validation_split=0.15)

model_qt = Model(inputs=input1, outputs=output1)

model_qt.compile(loss='mse', optimizer='adam')
model_qt.fit(x_train_qt, y_train, batch_size=13, epochs=300, verbose=0, validation_split=0.15)

loss_ma = model_ma.evaluate(x_test_ma, y_test)
predict_ma = model_ma.predict(x_test_ma)

loss_rb = model_rb.evaluate(x_test_rb, y_test)
predict_rb = model_rb.predict(x_test_rb)

loss_pt = model_pt.evaluate(x_test_pt, y_test)
predict_pt = model_pt.predict(x_test_pt)

loss_qt = model_qt.evaluate(x_test_qt, y_test)
predict_qt = model_qt.predict(x_test_qt)

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
  - loss: 16.838380813598633 r2: 0.8446910074225141
After scaling, hyperparameters are set identically
  for MaxAbsScaler,
   - loss: 98.17655944824219 accuracy: 0.09446743118442258
   - loss: 74.35376739501953 accuracy: 0.31419727021071164
   - loss: 83.6518325805664 accuracy: 0.22843642380948337
  for RobustScaler,
   - loss: 2839.639404296875 accuracy: -25.19144753709748
   - loss: 3730.278076171875 accuracy: -33.40626097124447
   - loss: 1696.7802734375 accuracy: -14.650272241816955
  for PowerTransformer,
   - loss: 3475.65380859375 accuracy: -31.057733015152884
   - loss: 3540.741455078125 accuracy: -31.658069146519246
   - loss: 3379.049560546875 accuracy: -30.166701434470077
  for QuantileTransformer,
   - loss: 14.3475923538208 accuracy: 0.8676648150082361
   - loss: 11.109149932861328 accuracy: 0.8975346251949831
   - loss: 12.613630294799805 accuracy: 0.8836580264760547
** implication: 
   - parameters are tuned according to un-scaled data, for that QuantileTransformer work fairly well.
     standard: slightly better accuracy
     minmax: slightly less accuracy
     maxabs, robust, power: greatly reduced accuracy
   - Since all the data are positive and min=0, minmax and maxabs should work the same, 
     but they don't, the result is very confusing for that reason.
     The data may be normaly distributed with huge standard deviation,
     or it may need a separate outlier clipping what ever it means.
   - un-scaled data's range
      for 0 column: 381.32 0.0
      for 1 column: 376.94 0.0
      for 2 column: 392.18 0.0
      for 3 column: 396.9 0.0
      for 4 column: 396.9 0.0
      for 5 column: 385.91 0.0
      for 6 column: 383.78 0.0
      for 7 column: 666.0 0.0
      for 8 column: 666.0 0.0
      for 9 column: 393.53 0.0
      for 10 column: 666.0 0.0
      for 11 column: 666.0 0.0
      for 12 column: 396.9 0.0
   - un-scaled data's standard deviation
      for 0 column: 117.63941133353315
      for 1 column: 121.5958944887161
      for 2 column: 119.57837495422027
      for 3 column: 116.59205747681169
      for 4 column: 125.66829045071947
      for 5 column: 123.45460137554572
      for 6 column: 121.11394817050487
      for 7 column: 194.2193273613109
      for 8 column: 189.97995409585678
      for 9 column: 120.94582832487406
      for 10 column: 190.74943032633738
      for 11 column: 194.07080825583913
      for 12 column: 125.85532182896176
'''