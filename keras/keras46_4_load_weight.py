from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import time

datasets = load_diabetes()

x = datasets.data   # (442, 10)
y = datasets.target # (442, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85)

model1 = load_model('./_save/keras46_3_save_weight_1.h5')

model = Sequential()
model.add(Dense(128 , input_shape=(10,)))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1))

start = time.time()
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.load_weights('./_save/keras46_3_save_weight_2.h5')
loss = model.evaluate(x_test, y_test)
predict = model.predict(x_test)

end = time.time() - start

r2 = r2_score(y_test, predict)

print('loss:', loss, 'actual data:', y_test, 'machine predicted:', predict)
print('accuracy:', r2)