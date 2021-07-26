from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Conv1D, GlobalAveragePooling1D, MaxPool1D, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=3600, test_split=0.2)
print(type(x_train), x_train.shape)
# <class 'numpy.ndarray'> (8982,)
print(type(y_train), y_train.shape)
# <class 'numpy.ndarray'> (8982,)
print(x_train[0], type(x_train[0]))
# indexes of words in sentence, <class 'list'>
print(len(x_train[0]), len(x_train[1]))
# 87, 56

dict_sentence = reuters.get_word_index()
print(type(reuters.get_word_index()))
# <class 'dict'>

# 문장의 길이
# [87, 56, 139, 224, 101, 116, 100, 100, 82, 106, 31, 59, 65, 316...]

# sentence = list(map(lambda x: dict_sentence.keys()[list(dict_sentence.values()).index(x)], dict_sentence))
# print(dict_sentence.keys()[[dict_sentence.values()].index(1)])
# print(list(map(lambda x: list(dict_sentence.keys())[list(dict_sentence.values()).index(x)], x_train[0])))

print('maximum lenth of articles:', max(len(i) for i in x_train))
# maximum lenth of articles: 2376
print('average length of articles:', sum(map(len, x_train)) / len(x_train))
# dict_sentence.keys()

plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

x_train = pad_sequences(x_train, maxlen=150, padding='pre')
x_test = pad_sequences(x_test, maxlen=150, padding='pre')
print(x_train.shape,type(x_train),x_train)
# (8982, 100) <class 'numpy.ndarray'> [[   0    0    0 ...   15   17   12]
#                                      [   0    0    0 ...  505   17   12]
#                                      [  19  758   15 ...   11   17   12]
#                                      ...
#                                      [   0    0    0 ...  407   17   12]
#                                      [  88 2321   72 ...  364   17   12]
#                                      [ 125 2855   21 ...  113   17   12]]

'''
y_train = set(y_train)
y_test = set(y_test)
print(y_test - y_train): 공집합
print(y_train - y_test): 공집합
'''

y_train = to_categorical(y_train)
# (8982, 46)
y_test = to_categorical(y_test)
# (2246, 46)

inputL = Input((150, ))
hl = Embedding(input_dim=3600, output_dim=32)(inputL)
hl = Conv1D(32, 2)(hl)
hl = Dropout(0.25)(hl)
hl = Conv1D(32, 2)(hl)
hl = MaxPool1D()(hl)
hl = GlobalAveragePooling1D()(hl)
outputL = Dense(46, activation='softmax')(hl)

model = Model(inputs=inputL, outputs=outputL)

print(np.unique(x_train))
print(np.unique(x_test))

es = EarlyStopping(monitor='val_acc', mode='max', patience=24)

start = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=240, batch_size=16, validation_split=0.2, verbose=2, callbacks=[es])
end = time.time() - start

acc = model.evaluate(x_test, y_test)[1]
print('time:',end,'seconds')
print('acc:', acc)

'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 100)]             0
_________________________________________________________________
embedding (Embedding)        (None, 100, 32)           896
_________________________________________________________________
conv1d (Conv1D)              (None, 99, 32)            2080
_________________________________________________________________
dropout (Dropout)            (None, 99, 32)            0
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 98, 32)            2080
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 49, 32)            0
_________________________________________________________________
global_average_pooling1d (Gl (None, 32)                0
_________________________________________________________________
dense (Dense)                (None, 46)                1518
=================================================================
Total params: 6,574
time: 2 minutes 16 seconds
acc: 0.7230632305145264
'''