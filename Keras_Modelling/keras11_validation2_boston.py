from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

datasets = load_boston()
x = datasets.data
y = datasets.target

# x.shape: (506, 13)
# y.shape: (506,)

# type(x): ndarray
# type(y): ndarray

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=47)

model = Sequential()
model.add(Dense(26, input_dim=13))
model.add(Dense(52))
model.add(Dense(13))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
start = time.time()
# bigger epoch size doesn't make whole lot of difference
model.fit(x_train, y_train, epochs=15000, batch_size=43, verbose=0, validation_split=0.3)

loss = model.evaluate(x_test, y_test)
y_predicted = model.predict(x_test)
r2 = r2_score(y_test, y_predicted)

end = time.time() - start

print('loss:', loss,'accuracy:',r2, 'took', end, 'seconds long')

'''
without validation in process
loss: 29.007150650024414 accuracy: 0.640081626141401 took 18.390971422195435 seconds long
with validation in process
loss: 31.38526725769043 accuracy: 0.6105741769732836 took 20.183785438537598 seconds long
loss: 61.88885498046875 accuracy: 0.2287743614102482 took 21.258233547210693 seconds long
raise epoch to be 2500
loss: 27.072874069213867 accuracy: 0.6626323864520285 took 62.9033567905426 seconds long
loss: 25.125600814819336 accuracy: 0.686898242066482 took 64.11098861694336 seconds long
raise epoch to be 5000
loss: 25.17871856689453 accuracy: 0.6862363160831505 took 126.78182625770569 seconds long
raise epoch to be 5000
loss: 23.340248107910156 accuracy: 0.7091463493523745 took 382.9617829322815 seconds long
'''