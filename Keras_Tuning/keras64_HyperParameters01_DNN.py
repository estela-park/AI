# A model built with tensorflow should be wrapped with sci-kit,
# then the model can be used as arg for sci-kit methods
# 
# tensorflow.wrapper(build_fn=function that builds a functional model)
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Data-prep
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28*28).astype('float32')
x_test = x_test.reshape(10000, 28*28).astype('float32')
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Modelling
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28), name='input')
    hl = Dense(512, activation='relu', name = 'hidden1')(inputs)
    hl = Dropout(drop)(hl)
    hl = Dense(256, activation='relu', name = 'hidden2')(hl)
    hl = Dropout(drop)(hl)
    hl = Dense(128, activation='relu', name = 'hidden3')(hl)
    hl = Dropout(drop)(hl)
    outputs = Dense(10, activation='softmax', name='outputs')(hl)
    model = Model(inputs, outputs)
    model.compile(optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model


def set_hyperparameter():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [.1, .2, .3]
    return {'batch_size': batches, 'optimizer': optimizers, 'drop': dropout}


model_precursor = KerasClassifier(build_fn=build_model)
# can give epochs in the wrapper
# can give validation_split in the wrapper
hyperparameter = set_hyperparameter()
model = GridSearchCV(model_precursor, hyperparameter, cv=5)
model.fit(x_train, y_train, verbose=1)
# can give epochs in fit fn
# can give validation_split in fit fn
print(model.score(x_test, y_test))


'''
**Default=[loss: 0.7466050982475281, accuracy: 0.8149999976158142]
**GridSearchCF=[loss: 0.2297, accuracy: 0.9336]
    > 0.9336000084877014
'''