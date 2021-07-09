# 1. R2를 음수가 아닌 0.5 이하로 만들기
# 2. 데이터는 건들지 않는다
# 3. 레이어 (including input layer and output layer) 갯수 6개 이상
# 4. batch_size=1
# 5. epochs >= 100
# 6. 10 <= the number of nodes <= 1000
# 7. train_size=0.7

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

x = np.array(range(100))
y = np.array(range(1, 101))


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(10))
model.add(Dense(1000))
model.add(Dense(10))
model.add(Dense(1000))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

loss = model.evaluate(x_test, y_test)
print('loss:', loss)
y_predict = model.predict([100])
print('prection for 100:', y_predict)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print('calculated accuracy(maximum_value=1):',r2)