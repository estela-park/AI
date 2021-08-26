from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Embedding, Input, Conv1D, MaxPool1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import time

from tensorflow.python.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=1000
)

# print(type(x_train), x_train.shape)    : <class 'numpy.ndarray'> (25000,)
# print(type(y_train), y_train.shape)    : <class 'numpy.ndarray'> (25000,)
# print(x_train[0], type(x_train[0]))    : [1, 14, 22, 16, 43, 530, ...32] <class 'list'>
# print(len(x_train[0]), len(x_train[1])): 218 189

dict_sentence = imdb.get_word_index()
# print(type(imdb.get_word_index())): <class 'dict'>
# print(dict_sentence.keys()): ['fawn', 'tsukino', 'nunnery', 'sonja', ...]
# print('maximum lenth of articles:', max(len(i) for i in x_train)): 2494
# print('average length of articles:', sum(map(len, x_train)) / len(x_train)): 238.71364

plt.hist([len(s) for s in x_train], bins=50)
# plt.show(): maxlen=350

x_train = pad_sequences(x_train, maxlen=350, padding='pre')
x_test = pad_sequences(x_test, maxlen=350, padding='pre')

# print(np.unique(x_train)) : [range(1000)]
# print(np.unique(x_test))  : [range(1000)]

# y_train = set(y_train)
# y_test = set(y_test)
# print(y_test - y_train): 공집합
# print(y_train - y_test): 공집합

inputL = Input((350, ))
hl = Embedding(input_dim=1000, output_dim=32)(inputL)
hl = Dropout(0.35)(hl)
hl = Conv1D(32, 2)(hl)
hl = MaxPool1D()(hl)
hl = GlobalAveragePooling1D()(hl)
outputL = Dense(1, activation='sigmoid')(hl)

model = Model(inputs=inputL, outputs=outputL)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc', mode='max', patience=6, verbose=2, restore_best_weights=True)

start = time.time()
hist = model.fit(x_train, y_train, batch_size=16, epochs=240, verbose=2, validation_split=0.15, callbacks=[es])
end = time.time() - start

acc = model.evaluate(x_test, y_test)[1]
print('time:',end,'seconds')
print('acc:', acc)

'''
First Model
-category
    time: 1 min 7 secs
    acc: 0.863399982452392
-binary
    time: 1 min 4 secs
    acc: 0.8626800179481506
    
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 350)]             0
_________________________________________________________________
embedding (Embedding)        (None, 350, 32)           32000
_________________________________________________________________
dropout (Dropout)            (None, 350, 32)           0
_________________________________________________________________
conv1d (Conv1D)              (None, 349, 32)           2080
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 174, 32)           0
_________________________________________________________________
global_average_pooling1d (Gl (None, 32)                0
_________________________________________________________________
dense (Dense)                (None, 2)                 66
=================================================================
Total params: 34,146
'''