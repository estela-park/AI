import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV

# Data-prep
datasets = load_wine() 
y = to_categorical(datasets.target)
x_train, x_test, y_train, y_test = train_test_split(datasets.data, y, random_state=72)
# (133, 13) (45, 13) (133, 3) (45, 3)


# Modelling
def build_model(learning_rate=0.001, units_first=512, units_last=128, batches=100):
    inputs = Input(shape=(13), name='input')
    hl = Dense(units_first, activation='relu', name = 'hidden1')(inputs)
    hl = Dropout(.25)(hl)
    hl = Dense(256, activation='relu', name = 'hidden2')(hl)
    hl = Dropout(.25)(hl)
    hl = Dense(units_last, activation='relu', name = 'hidden3')(hl)
    hl = Dropout(.25)(hl)
    outputs = Dense(3, activation='softmax', name='outputs')(hl)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate), metrics=['acc'], loss='categorical_crossentropy')
    return model


def set_hyperparameter():
    learning_rate = [0.001, 0.005, 0.01]
    units_first = [512, 256]
    units_last = [128, 64, 32]
    dropout = [.15, .25]
    batches = [100, 500]
    return {'learning_rate':learning_rate,'units_first': units_first,'units_last': units_last,
            'batch_size': batches,}


model_precursor = KerasClassifier(build_fn=build_model)
hyperparameter = set_hyperparameter()
model = GridSearchCV(model_precursor, hyperparameter, cv=2)
start = time.time()
model.fit(x_train, y_train, verbose=2)
end = time.time() - start
print(model.score(x_test, y_test))
print(end//60, ' ', end%60)

'''
**Default Model: 0.7 seconds
    > [loss: 8.3466 - acc: 0.3111]

**GridSearch: 33 seconds
    > [loss: 16.5052 - acc: 0.3111]
'''