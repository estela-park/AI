from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Data 

x1 = np.array([range(100), range(301, 401), range(1, 101)])
x1 = np.transpose(x1)
y1 = np.array([range(1001, 1101)])
y2 = np.array(range(1901, 2001))
y1 = np.transpose(y1)
print(x1.shape, y1.shape)

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, y1, y2, test_size=0.3, random_state=8, shuffle=True)


# Modelling

#2-1 분기 이전 모델
input1 = Input(shape=(3,))
dense1 = Dense(55, activation='relu', name='dense1')(input1)
dense2 = Dense(32, activation='relu', name='dense2')(dense1)
dense3 = Dense(26, activation='relu', name='dense3')(dense2)
output1 = Dense(18)(dense3)

#2-2 분기 이후 모델

modelense1 = Dense(24)(output1)
modelense2 = Dense(24)(modelense1)
modelense3 = Dense(24)(modelense2)
modelense4 = Dense(24)(modelense3)
output21 = Dense(7)(modelense4)

modelense11 = Dense(24)(output1)
modelense12 = Dense(24)(modelense11)
modelense13 = Dense(24)(modelense12)
modelense14 = Dense(24)(modelense13)
output22 = Dense(8)(modelense14)

last_output1 = Dense(1, name='outputdense1')(output21)
last_output2 = Dense(1, name='outputdense2')(output22)

model = Model(inputs=input1, outputs=[last_output1, last_output2])

model.summary()


# 3. Compilation, Training
model.compile(loss = 'mse', optimizer='adam', metrics=['mae']) 
model.fit(x1_train, [y1_train, y2_train], epochs=100, batch_size=8, verbose=1, validation_split=0.1)

# 4. Evaluation, Prediction
result = model.evaluate(x1_test, [y1_test, y2_test])
print('result : ', result)
y_predict = model.predict(x1_test)

print('loss : ', result[0])
print('metrics["mae"] : ', result[1])