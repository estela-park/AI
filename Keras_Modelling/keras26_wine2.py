import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

dataset = pd.read_csv('../_data/winequality-white.csv', sep=';', header=0)

x = dataset.iloc[:, :11] # (4898, 11), DataFrame
x = np.array(x)
y = dataset['quality']   # (4898, )  , Series   , [6 5 7 8 4 3 9]
y = np.array(y)          # [6 6 6 6 6 6 6 6 6 6 5 5 5 7 5 7 6 8 6 5], (4898, )
# y = np.array(y - 3)      # 3-9 -> 0-6
# y = to_categorical(y)    # (4849, 10)

y = y.reshape(4898, 1)
enc = OneHotEncoder(handle_unknown='ignore')
y = enc.fit_transform(y) # <class 'scipy.sparse.csr.csr_matrix'>
y = y.toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=72)

standard_scaler = StandardScaler()
standard_scaler.fit(x_train)
x_train_std = standard_scaler.transform(x_train)
x_test_std = standard_scaler.transform(x_test)
'''
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train)
x_train_mm = min_max_scaler.transform(x_train)
x_test_mm = min_max_scaler.transform(x_test)

max_abs_scaler = MaxAbsScaler()
max_abs_scaler.fit(x_train)
x_train_ma = max_abs_scaler.transform(x_train)
x_test_ma = max_abs_scaler.transform(x_test)
'''
robust_scaler = RobustScaler()
robust_scaler.fit(x_train)
x_train_rb = robust_scaler.transform(x_train)
x_test_rb = robust_scaler.transform(x_test)

power_scaler = PowerTransformer()
power_scaler.fit(x_train)
x_train_pt = power_scaler.transform(x_train)
x_test_pt = power_scaler.transform(x_test)
'''
quantile_scaler = QuantileTransformer()
quantile_scaler.fit(x_train)
x_train_qt = quantile_scaler.transform(x_train)
x_test_qt = quantile_scaler.transform(x_test)
'''
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy', patience=15, mode='max', verbose=2)

start = time.time()
'''
model = Sequential()
model.add(Dense(64, input_shape=(11,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])
'''
model_std = Sequential()
model_std.add(Dense(64, input_shape=(11,), activation='relu'))
model_std.add(Dense(128, activation='relu'))
model_std.add(Dense(32, activation='relu'))
model_std.add(Dense(7, activation='softmax'))

model_std.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_std.fit(x_train_std, y_train, epochs=360, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])
'''
model_mm = Sequential()
model_mm.add(Dense(64, input_shape=(11,), activation='relu'))
model_mm.add(Dense(128, activation='relu'))
model_mm.add(Dense(32, activation='relu'))
model_mm.add(Dense(7, activation='softmax'))

model_mm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_mm.fit(x_train_mm, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

model_ma = Sequential()
model_ma.add(Dense(64, input_shape=(11,), activation='relu'))
model_ma.add(Dense(128, activation='relu'))
model_ma.add(Dense(32, activation='relu'))
model_ma.add(Dense(7, activation='softmax'))

model_ma.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ma.fit(x_train_ma, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])
'''
model_rb = Sequential()
model_rb.add(Dense(64, input_shape=(11,), activation='relu'))
model_rb.add(Dense(128, activation='relu'))
model_rb.add(Dense(32, activation='relu'))
model_rb.add(Dense(7, activation='softmax'))

model_rb.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_rb.fit(x_train_rb, y_train, epochs=360, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])

model_pt = Sequential()
model_pt.add(Dense(64, input_shape=(11,), activation='relu'))
model_pt.add(Dense(128, activation='relu'))
model_pt.add(Dense(32, activation='relu'))
model_pt.add(Dense(7, activation='softmax'))

model_pt.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_pt.fit(x_train_pt, y_train, epochs=360, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])
'''
model_qt = Sequential()
model_qt.add(Dense(64, input_shape=(11,), activation='relu'))
model_qt.add(Dense(128, activation='relu'))
model_qt.add(Dense(32, activation='relu'))
model_qt.add(Dense(7, activation='softmax'))

model_qt.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_qt.fit(x_train_qt, y_train, epochs=240, validation_split=0.15, verbose=2, batch_size=32, callbacks=[es])
'''
# loss = model.evaluate(x_test, y_test)
# predict = model.predict(x_test)

loss_std = model_std.evaluate(x_test_std, y_test)
predict_std = model_std.predict(x_test_std)

# loss_mm = model_mm.evaluate(x_test_mm, y_test)
# predict_mm = model_mm.predict(x_test_mm)

# loss_ma = model_ma.evaluate(x_test_ma, y_test)
# predict_ma = model_ma.predict(x_test_ma)

loss_rb = model_rb.evaluate(x_test_rb, y_test)
predict_rb = model_rb.predict(x_test_rb)

loss_pt = model_pt.evaluate(x_test_pt, y_test)
predict_pt = model_pt.predict(x_test_pt)

# loss_qt = model_qt.evaluate(x_test_qt, y_test)
# predict_qt = model_qt.predict(x_test_qt)

end = time.time() - start

print('it took', end/60,'minutes and', end%60, 'seconds')

# print('without scaling, accuracy:', loss)
print('for stdS, accuracy:', loss_std)
# print('for minmax, accuracy:', loss_mm)
# print('for maxabs, accuracy:', loss_ma)
print('for robust, accuracy:', loss_rb)
print('for powerT, accuracy:', loss_pt)
# print('for quantileT, accuracy:', loss_qt)

'''
it took 338.4814522266388 seconds
without scaling, accuracy: [1.234687089920044, 0.5346938967704773]
for stdS, accuracy: [3.2737958431243896, 0.6081632375717163]
for minmax, accuracy: [1.453718900680542, 0.5850340127944946]
for maxabs, accuracy: [1.1606807708740234, 0.5523809790611267]
for robust, accuracy: [2.8314061164855957, 0.6000000238418579]
for powerT, accuracy: [2.523308277130127, 0.6095238327980042]
for quantileT, accuracy: [1.9040074348449707, 0.6068027019500732]
-
it took 5.922241655985514 minutes and 55.33449935913086 seconds
without scaling, accuracy: [1.0809168815612793, 0.5401360392570496]
for stdS, accuracy: [2.990638256072998, 0.5904762148857117]
for minmax, accuracy: [1.0695478916168213, 0.5918367505073547]
for maxabs, accuracy: [1.0419076681137085, 0.5714285969734192]
for robust, accuracy: [2.818464994430542, 0.6122449040412903]
for powerT, accuracy: [2.7235107421875, 0.6149659752845764]
for quantileT, accuracy: [2.0557126998901367, 0.5809524059295654]
-
without scaling, accuracy: [1.3597968816757202, 0.525170087814331]
for stdS, accuracy: [2.8630237579345703, 0.601360559463501]
for minmax, accuracy: [1.3435444831848145, 0.5700680017471313]
for maxabs, accuracy: [1.107465147972107, 0.5673469305038452]
for robust, accuracy: [3.192901611328125, 0.6068027019500732]
for powerT, accuracy: [3.150344133377075, 0.601360559463501]
for quantileT, accuracy: [2.102902412414551, 0.5918367505073547]
- with higher output scaler
it took 1.8676807284355164 minutes and 52.06084370613098 seconds
for stdS, accuracy: [2.875603437423706, 0.6217687129974365]
for robust, accuracy: [2.343806743621826, 0.6272108554840088]
for powerT, accuracy: [2.84159517288208, 0.646258533000946]
- model stopped too early, altered patience
it took 2.424021311601003 minutes and 25.44127869606018 seconds
for stdS, accuracy: [3.0749671459198, 0.6272108554840088]
for robust, accuracy: [3.2979772090911865, 0.6448979377746582]
for powerT, accuracy: [3.2596006393432617, 0.6503401398658752]
-
it took 2.754525963465373 minutes and 45.27155780792236 seconds
for stdS, accuracy: [3.392552614212036, 0.6231292486190796]
for robust, accuracy: [3.6813735961914062, 0.6190476417541504]
for powerT, accuracy: [4.0151262283325195, 0.6149659752845764]
-still too early
it took 2.853236754735311 minutes and 51.19420528411865 seconds
for stdS, accuracy: [3.575083017349243, 0.6448979377746582]
for robust, accuracy: [2.9790892601013184, 0.636734664440155]
for powerT, accuracy: [4.102970123291016, 0.6394557952880859]
-altered early stopping to watch validation's accuracy
it took 0.4712220788002014 minutes and 28.273324728012085 seconds
for stdS, accuracy: [1.157938003540039, 0.5659863948822021]
for robust, accuracy: [1.1357967853546143, 0.5687074661254883]
for powerT, accuracy: [1.1126573085784912, 0.5931972861289978]
-watching for train's accuracy
it took 2.426863519350688 minutes and 25.61181116104126 seconds
for stdS, accuracy: [3.30517840385437, 0.6204081773757935]
for robust, accuracy: [3.286365032196045, 0.6489796042442322]
for powerT, accuracy: [3.4653685092926025, 0.6693877577781677]
-
it took 2.698052167892456 minutes and 41.88313007354736 seconds
for stdS, accuracy: [2.9447574615478516, 0.6435374021530151]
for robust, accuracy: [3.681108236312866, 0.6204081773757935]
for powerT, accuracy: [3.3715384006500244, 0.636734664440155]
- with OneHotEncoder
it took 2.1935532967249554 minutes and 11.613197803497314 seconds
for stdS, accuracy: [2.9117071628570557, 0.6176870465278625]
for robust, accuracy: [2.817774772644043, 0.635374128818512]
for powerT, accuracy: [3.1451005935668945, 0.6217687129974365]
-
it took 1.992166554927826 minutes and 59.529993295669556 seconds
for stdS, accuracy: [2.9271061420440674, 0.6340135931968689]
for robust, accuracy: [2.706691026687622, 0.5836734771728516]
for powerT, accuracy: [2.5320143699645996, 0.6421768665313721]
'''