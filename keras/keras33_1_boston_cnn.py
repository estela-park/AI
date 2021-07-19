from sklearn.datasets import load_boston
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, Dropout, GlobalAveragePooling2D, Layer, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import QuantileTransformer
import time

dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=82)

quantile_scaler = QuantileTransformer()
x_train = quantile_scaler.fit_transform(x_train)
x_test = quantile_scaler.transform(x_test)
# (430, 13) (76, 13)

x_train = x_train.reshape(430, 1, 1, 13)
x_test = x_test.reshape(76, 1, 1, 13)

input_l = Input(shape=(1, 1, 13))
hl = Conv2D(filters=104, kernel_size=(1, 1), padding='same', activation='relu')(input_l)
hl = Dropout(0.2)(hl)
hl = Conv2D(52, (1, 1), activation='relu')(hl)
hl = Dropout(0.2)(hl)
hl = Conv2D(26, (1, 1), activation='relu')(hl)
hl = Dropout(0.2)(hl)
hl = Conv2D(13, (1, 1), activation='relu')(hl)
hl = Flatten()(hl)
output_l = Dense(1)(hl)

model = Model(inputs=[input_l], outputs=[output_l])

model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', patience=24, mode='min', verbose=2)

start = time.time()
model.fit(x_train, y_train, batch_size=13, epochs=300, verbose=2, validation_split=0.15)
end = time.time() - start

loss = model.evaluate(x_test, y_test)
predict = model.predict(x_test)

r2 = r2_score(y_test, predict)

print('loss:',loss,'accuracy:',r2)

'''
CNN [epochs=300, batch_size=48]
    loss: 14.8783540725708 accuracy: 0.8627693282994542
    [epochs=300, batch_size=13]
    loss: 13.280355453491211 accuracy: 0.8775084879017759
DNN [epochs=300, batch_size=13]
    loss: 11.109149932861328 accuracy: 0.8975346251949831
'''