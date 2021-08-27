import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Data-prep
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Modelling
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28, 28, 1), name='input')
    hl = Conv2D(16, (2, 2), activation='relu', name = 'hiddenC1')(inputs)
    hl = Dropout(drop)(hl)
    hl = Conv2D(4, (2, 2), activation='relu', name = 'hiddenC2')(hl)
    hl = MaxPool2D()(hl)
    hl = Flatten()(hl)
    hl = Dense(512, activation='relu', name = 'hiddenD1')(hl)
    hl = Dropout(drop)(hl)
    hl = Dense(128, activation='relu', name = 'hiddenD2')(hl)
    hl = Dropout(drop)(hl)
    hl = Dense(32, activation='relu', name = 'hiddenD3')(hl)
    hl = Dropout(drop)(hl)
    outputs = Dense(10, activation='softmax', name='outputs')(hl)
    model = Model(inputs, outputs)
    model.compile(optimizer, metrics=['acc'], loss='categorical_crossentropy')
    print('**********************************')
    return model


'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           [(None, 28, 28, 1)]       0
_________________________________________________________________
hiddenC1 (Conv2D)            (None, 27, 27, 16)        80
_________________________________________________________________
dropout (Dropout)            (None, 27, 27, 16)        0
_________________________________________________________________
hiddenC2 (Conv2D)            (None, 26, 26, 4)         260
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 4)         0
_________________________________________________________________
flatten (Flatten)            (None, 676)               0
_________________________________________________________________
hiddenD1 (Dense)             (None, 512)               346624
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0
_________________________________________________________________
hiddenD2 (Dense)             (None, 128)               65664
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0
_________________________________________________________________
hiddenD3 (Dense)             (None, 32)                4128
_________________________________________________________________
dropout_3 (Dropout)          (None, 32)                0
_________________________________________________________________
outputs (Dense)              (None, 10)                330
=================================================================
Total params: 417,086
'''


def set_hyperparameter():
    batches = [20, 30, 40]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [.2, .3]
    return {'batch_size': batches, 'optimizer': optimizers, 'drop': dropout}


model_precursor = KerasClassifier(build_fn=build_model)
hyperparameter = set_hyperparameter()
model = GridSearchCV(model_precursor, hyperparameter, cv=2)
start = time.time()
model.fit(x_train, y_train, verbose=2)
end = time.time() - start
print(end//60, ' ', end%60)
print(model.score(x_test, y_test))


'''
**Default=[loss: 0.7466050982475281, accuracy: 0.8149999976158142]
**GridSearchCV_DNN=[loss: 0.2297, accuracy: 0.9336]
**GridSearchCV_CNN=[loss: 0.1538 - acc: 0.9573]

CNN_GridSearchCV(iter=36) took 4 minutes and 51 seconds
'''