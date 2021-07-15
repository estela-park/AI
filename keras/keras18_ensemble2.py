import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
from sklearn.model_selection import train_test_split

x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)

y1 = np.array(range(1001, 1101))
y2 = np.array(range(1901, 2001))
# x1.shape: (100, 3), x2.shape: (100, 3), y1.shape: (100,)

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, x2, y1, y2, train_size=0.85, random_state=74)

input11 = Input(shape=(3, ))
dense11 = Dense(12, activation='relu', name='HiddenA1')(input11)
dense12 = Dense(6, activation='relu', name='HiddenA2')(dense11)
dense13 = Dense(3, activation='relu', name='HiddenA3')(dense12)
output11 = Dense(1, name='HiddenA4')(dense13)

input21 = Input(shape=(3, ))
dense21 = Dense(12, activation='relu', name='HiddenB1')(input21)
dense22 = Dense(6, activation='relu', name='HiddenB2')(dense21)
dense23 = Dense(3, activation='relu', name='HiddenB3')(dense22)
output21 = Dense(1, name='HiddenB4')(dense23)

merge1 = concatenate([output11, output21], name='Merged1')
merge2 = Dense(10, name='Merged2')(merge1)
merge3 = Dense(5, activation='relu', name='Merged3')(merge2)

output12 = Dense(8, name='BranchedA1')(merge3)
last_output1 = Dense(1, name='BranchedA2')(output12)

output22 = Dense(8, name='BranchedB1')(merge3)
last_output2 = Dense(1, name='BranchedB2')(output22)

model = Model(inputs=[input11, input21], outputs=[last_output1, last_output2])

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], [y1_train, y2_train], batch_size=8, epochs=160, verbose=0, validation_split=0.15)

result = model.evaluate([x1_test, x2_test], [y1_test, y2_test])
print(result)
loss = result[0]
mae = result[1]
print('loss:', loss, 'mae', mae)

predict = model.predict([x1_test, x2_test])