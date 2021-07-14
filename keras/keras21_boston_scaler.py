from sklearn.datasets import load_boston
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=82)

input1 = Input(shape=(13, ))
dense1 = Dense(104, activation='relu')(input1)
dense2 = Dense(52, activation='relu')(dense1)
dense3 = Dense(26, activation='relu')(dense2)
dense4 = Dense(13, activation='relu')(dense3)
output1 = Dense(1)(dense4)

model = Model(inputs=input1, outputs=output1)

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size=13, epochs=300, verbose=0, validation_split=0.15)

loss = model.evaluate(x_test, y_test)
predict = model.predict(x_test)
r2 = r2_score(y_test, predict)

print('loss:', loss, 'r2:', r2)

'''
Before scaling,
  - loss: 16.838380813598633 r2: 0.8446910074225141
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