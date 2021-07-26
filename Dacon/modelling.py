import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
import time
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Conv1D, GlobalAveragePooling1D, MaxPool1D, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

train = pd.read_csv('./Dacon/_data/train_data.csv', encoding='utf-8')
train = train.set_index('index')
x = train['title']
# Series
y = train['topic_idx']

sub = pd.read_csv('./Dacon/_data/test_data.csv', encoding='utf-8')
sub = sub.set_index('index')
sub_x = sub['title']

text = ' '.join(x)
text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", text)
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
# tokenizer.word_index: <class 'dict'> 76528

x = list(x)
for i, j in enumerate(x):
    text_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", j)
    x[i] = text_clean

sub_x = list(sub_x)
for i, j in enumerate(sub_x):
    text_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", j)
    sub_x[i] = text_clean

x = tokenizer.texts_to_sequences(x)
sub_x = tokenizer.texts_to_sequences(sub_x)

x = pad_sequences(x, maxlen=15, padding='pre')
sub_x = pad_sequences(sub_x, maxlen=15, padding='pre')
y = to_categorical(y)

train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.75)
# (34240, 7) (11414, 7) (34240, 7)(11414, 7)

inputL = Input((15, ))
hl = Embedding(input_dim=76529, output_dim=128)(inputL)
hl = Conv1D(128, 2, activation='relu')(hl)
hl = Dropout(0.85)(hl)
hl = Conv1D(64, 2, activation='relu')(hl)
hl = MaxPool1D()(hl)
hl = GlobalAveragePooling1D()(hl)
outputL = Dense(7, activation='softmax')(hl)

model = Model(inputs=inputL, outputs=outputL)

model.summary()

date = datetime.datetime.now()
date_time = date.strftime('%m%d-%H%M')
f_path = f'./Dacon/_saveW/test_{date_time}' + '_{epoch:04d}_{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_acc', mode='max', patience=5, verbose=2, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_acc', mode='max', verbose=2, save_best_only=True, filepath=f_path)

start = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(train_x, train_y, epochs=360, batch_size=16, validation_split=0.75, verbose=2, callbacks=[es, mcp])
end = time.time() - start

acc = model.evaluate(test_x, test_y)
temp = model.predict(sub_x)
# (9131, 7)
index = tf.argmax(temp, axis=1)
# [0 1 2 3 4 5 6] <class 'tensorflow.python.framework.ops.EagerTensor'>
df = pd.DataFrame(index)
df.to_csv('./Dacon/_save/pre_submission.csv')

print('time:',end,'seconds')
print('acc:', acc)

'''
First Model
time: 6 mins 9 secs
acc: [0.45428329706192017, 0.7531102299690247]
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 10)]              0
_________________________________________________________________
embedding (Embedding)        (None, 10, 32)            2448928
_________________________________________________________________
conv1d (Conv1D)              (None, 9, 64)             4160
_________________________________________________________________
dropout (Dropout)            (None, 9, 64)             0
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 8, 32)             4128
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 4, 32)             0
_________________________________________________________________
global_average_pooling1d (Gl (None, 32)                0
_________________________________________________________________
dense (Dense)                (None, 7)                 231
=================================================================
Total params: 2,457,447
'''