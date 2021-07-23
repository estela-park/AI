from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Dropout, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import time

datasets = load_diabetes()

x = datasets.data   
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=72)
# (353, 10) (89, 10) (353,) (89,)

quantile_scaler = QuantileTransformer()
x_train_qt = quantile_scaler.fit_transform(x_train)
x_test_qt = quantile_scaler.transform(x_test)

x_train = x_train.reshape(353, 10, 1)
x_test = x_test.reshape(89, 10, 1)
x_train_qt = x_train_qt.reshape(353, 10, 1)
x_test_qt = x_test_qt.reshape(89, 10, 1)

inputL = Input(shape=(10, 1))
hl = LSTM(12, activation='relu', return_sequences=True)(inputL)
hl = Dropout(0.2)(hl)
hl = Conv1D(128, 3, activation='relu')(hl)
hl = MaxPooling1D()(hl)
hl = Conv1D(32, 2, activation='relu')(hl)
hl = Dropout(0.2)(hl)
hl = Flatten()(hl)
outputL = Dense(1)(hl)

model = Model(inputs=[inputL], outputs=[outputL])
model_qt = Model(inputs=[inputL], outputs=[outputL])
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model_qt.compile(loss='mse', optimizer='adam', metrics=['mae'])

es = EarlyStopping(monitor='loss', mode='min', patience=24, verbose=2)

start = time.time()
model.fit(x_train, y_train, epochs=360, batch_size=8, validation_split=0.15, callbacks=[es])
model_qt.fit(x_train_qt, y_train, epochs=360, batch_size=8, validation_split=0.15, callbacks=[es])
end = time.time() - start

loss = model.evaluate(x_test, y_test)
predict = model.predict(x_test)


loss_qt = model_qt.evaluate(x_test_qt, y_test)
predict_qt = model_qt.predict(x_test_qt)

r2 = r2_score(y_test, predict)
r2_qt = r2_score(y_test, predict_qt)

print('it took', end/60, 'minutes and', end%60,'seconds')
print('without scaling, loss:', loss, 'accuracy:', r2)
print('for quantile transformer, loss:', loss_qt, 'accuracy:', r2_qt)