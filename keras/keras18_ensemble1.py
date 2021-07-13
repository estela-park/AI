import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
from sklearn.model_selection import train_test_split

x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)

y1 = np.array([range(1001, 1101)])
y1 = np.transpose(y1)

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, train_size=0.85)

input11 = Input(shape=(3, ))
dense11 = Dense(48, activation='relu')(input11)
dense12 = Dense(12, activation='relu')(dense11)
dense13 = Dense(3, activation='relu')(dense12)
output11 = Dense(1)(dense13)


input21 = Input(shape=(3, ))
dense21 = Dense(10, activation='relu')(input21)
dense22 = Dense(10, activation='relu')(dense21)
dense23 = Dense(10, activation='relu')(dense22)
output21 = Dense(1)(dense23)

merge1 = concatenate([output11, output21])
merge2 = Dense(10, name='hidden_altered1')(merge1)
merge3 = Dense(5, activation='relu', name='altered2')(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input11, input21], outputs=last_output)

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], y1, batch_size=8, epochs=160, verbose=0, validation_split=0.15)

# how model.evaluate's metrics works

result = model.evaluate([x1_test, x2_test], y1_test)
loss = result[0]
mae = result[1]
print('loss:', loss, 'mae', mae)

predict = model.predict([x1_test, x2_test])
print(predict.shape)
print(predict)

print(y1_test.shape)
print(y1_test)