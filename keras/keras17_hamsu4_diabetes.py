from re import A
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import time

datasets = load_diabetes()

x = datasets.data   
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=72)

start = time.time()

input1 = Input(shape=(10,))
dense1 = Dense(64, activation='relu')(input1)
dense2 = Dense(128, activation='relu')
dense3 = Dense(32, activation='relu')
output1 = Dense(1)(dense3)

model = Model(inputs=input1, outputs=output1)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=240, validation_split=0.15, verbose=0, batch_size=32)

loss = model.evaluate(x_test, y_test)
predict = model.predict(x_test)

end = time.time() - start

r2 = r2_score(y_test, predict)

print('loss:', loss, 'time took:', end)
print('accuracy:', r2)