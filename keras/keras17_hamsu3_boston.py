from sklearn.datasets import load_boston
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=82)

input1 = Input(shape=(13, ))
hl = Dense(128, activation='relu')(input1)
hl = Dense(64, activation='relu')(hl)
hl = Dense(64, activation='relu')(hl)
hl = Dense(64, activation='relu')(hl)
hl = Dense(32, activation='relu')(hl)
output1 = Dense(1)(hl)

model = Model(inputs=input1, outputs=output1)

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size=13, epochs=300, verbose=0, validation_split=0.15)

loss = model.evaluate(x_test, y_test)
predict = model.predict(x_test)
r2 = r2_score(y_test, predict)

print('loss:', loss, 'r2:', r2)

'''
shape of the model to be cylinderical,
epochs=300, w/h activation fn
loss: 15.694770812988281 r2: 0.8552390879201758
loss: 16.560565948486328 r2: 0.8472534301151344

epochs=300, w/h activation fn
loss: 16.838380813598633 r2: 0.8446910074225141

epochs=3200, w/h activation fn
loss: 11.550013542175293 r2: 0.8934683240263458

epochs=10000, w/h activation fn
loss: 20.832361221313477 r2: 0.8078524758174028

epochs=300, w/o activation fn
loss: 24.207006454467773 r2: 0.7767264253621938
'''

'''
for random_state
4 done
loss: 27.2264404296875 r2: 0.6691044297937997, 0.67
22 done
loss: 34.87919616699219 r2: 0.671437213151485
31 done
loss: 20.12615394592285 r2: 0.7529869582052349, 0.69, 0.74
35 done
loss: 21.440065383911133 r2: 0.661353686153004
38 done
loss: 24.7960205078125 r2: 0.6527754966408476
40 done
loss: 43.2165412902832 r2: 0.6749945332631195
43 done
loss: 30.97711181640625 r2: 0.6777240638345554, 0.69
44 done
loss: 24.787521362304688 r2: 0.6893541485609288
46 done
loss: 16.742584228515625 r2: 0.750755772344852
48 done
loss: 25.449050903320312 r2: 0.6849952678312949, 0.66, 0.71
49 done
loss: 28.55120277404785 r2: 0.664333425434332
55 done
loss: 18.251691818237305 r2: 0.761262387175949, 0.69, 0.74, 0.73, 0.73, 0.74
58 done
loss: 31.257850646972656 r2: 0.6757975263871621
61 done
loss: 34.877593994140625 r2: 0.6620749682578329
70 done
loss: 30.81463050842285 r2: 0.6719711553765884
80 done
loss: 19.262155532836914 r2: 0.6778502323568459
82 done
loss: 24.992258071899414 r2: 0.7694836127660677, 0.73, 0.70, 0.75, 0.75
86 done
loss: 33.303504943847656 r2: 0.6563967995379745
87 done
loss: 34.02724838256836 r2: 0.6601505169027475
93 done
loss: 24.254980087280273 r2: 0.6658868501401911
94 done
loss: 29.431119918823242 r2: 0.6622270806617525
96 done
loss: 23.033273696899414 r2: 0.6947792185122903, 0.71, 0.70, 0.71
99 done
loss: 27.76909828186035 r2: 0.6747480574273867, 0.69
100 done
loss: 18.492483139038086 r2: 0.7429270700046704
102 done
loss: 26.39979362487793 r2: 0.6806118682524489, 0.68, 0.66
109 done
loss: 15.533473014831543 r2: 0.772806640701482, 0.71, 0.72, 0.74, 0.72
114 done
loss: 26.767602920532227 r2: 0.6566047881846598, 0.65, 0.65
125 done
loss: 35.529823303222656 r2: 0.6692095435985245
134 done
loss: 23.319156646728516 r2: 0.7099591471515496, 0.67
137 done
loss: 20.84256935119629 r2: 0.7427588622943031, 0.71, 0.73, -
139 done
loss: 32.67340850830078 r2: 0.680774449382245
144 done
loss: 24.58257484436035 r2: 0.7346112045058782, 0.73, 0.71, 0.66, 0.73
149 done
loss: 16.502456665039062 r2: 0.765294825011214, 0.72, 0.71, 0.68, -
'''