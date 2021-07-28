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
from tensorflow.keras.layers import Embedding, Input, Conv1D, GlobalAveragePooling1D, MaxPool1D, Dropout, Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

def func(text):
    clean = []
    for word in okt.pos(text, stem=True): #어간 추출
        if word[1] not in ['Josa', 'Eomi', 'Punctuation']: #조사, 어미, 구두점 제외 
            clean.append(word[0])
    return " ".join(clean) 

train = pd.read_csv('./Dacon/_data/train_data.csv', encoding='utf-8')
train = train.set_index('index')
x = train['title']

# Series
y = train['topic_idx']

sub = pd.read_csv('./Dacon/_data/test_data.csv', encoding='utf-8')
sub = sub.set_index('index')
sub_x = sub['title']

tokenizer = Tokenizer()

text = ' '.join(x)
text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", text)
tokenizer.fit_on_texts([text])
print(tokenizer.word_index)
# text = func(text)
tokenizer.fit_on_texts([text])
# text = func(text)
# tokenizer.word_index: <class 'dict'> 76528

# train['title'] = train['title'].apply(lambda x : func(x))

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

x = pad_sequences(x, maxlen=10, padding='pre')
sub_x = pad_sequences(sub_x, maxlen=10, padding='pre')
y = to_categorical(y)

train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.75, random_state=80)
# (34240, 7) (11414, 7) (34240, 7) (11414, 7)

inputL = Input((10, ))
hl = Embedding(input_dim=76529, output_dim=32)(inputL)
hl = LSTM(64, activation='relu', return_sequences=True)(hl)
hl = Conv1D(32, 2, activation='relu')(hl)
hl = Dropout(0.45)(hl)
hl = MaxPool1D()(hl)
hl = GlobalAveragePooling1D()(hl)
outputL = Dense(7, activation='softmax')(hl)

model = Model(inputs=inputL, outputs=outputL)

model.summary()

date = datetime.datetime.now()
date_time = date.strftime('%m%d-%H%M')
f_path = f'./Dacon/_saveW/{date_time}' + '_{epoch:04d}_{val_loss:.4f}.hdf5'
print('it will be saved at', date_time)

es = EarlyStopping(monitor='val_acc', mode='max', patience=6, verbose=2, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_acc', mode='max', verbose=0, save_best_only=True, filepath=f_path)

start = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(train_x, train_y, epochs=72, batch_size=16, validation_split=0.75, verbose=2, callbacks=[es, mcp])
end = time.time() - start

acc = model.evaluate(test_x, test_y)
temp = model.predict(sub_x)
# (9131, 7)
temp = tf.argmax(temp, axis=1)
# [0 1 2 3 4 5 6] <class 'tensorflow.python.framework.ops.EagerTensor'>
df = pd.DataFrame(temp)
df['index'] = np.array(range(45654, 54785))
df.rename(columns={0:'topic_idx'}, inplace=True)
df = df.set_index('index')
f_path = f'./Dacon/_save/{acc[1]}_{i}_{date_time}.csv'
df.to_csv(f_path)

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

Second Model
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 15)]              0
_________________________________________________________________
embedding (Embedding)        (None, 15, 49)            3749921
_________________________________________________________________
lstm (LSTM)                  (None, 14)                3584
_________________________________________________________________
dropout (Dropout)            (None, 14)                0
_________________________________________________________________
dense (Dense)                (None, 7)                 105
=================================================================
Total params: 3,753,610
output_dim=49, hiddenState=(14x1), Dropout(0.85): around 0.65   7:2   -acc < val_acc
output_dim=98, hiddenState=(14x1), Dropout(0.45): around 0.65   14:2  -acc < val_acc
output_dim=49, hiddenState=(156x1), Dropout(0.45): around 0.715 1:4   -overfitting
output_dim=24, hiddenState=(156x1), Dropout(0.45): around 0.715 1:7.5 -overfitting lessor degree
output_dim=98, hiddenState=(156x1), Dropout(0.45): around 0.73  1:2   -overfitting lessor degree
output_dim=49, hiddenState=(78x1), Dropout(0.45): around 0.725  1:2   -overfitting more lessor degree
output_dim=78, hiddenState=(78x1), Dropout(0.45): around 0.715  1:1   -overfitting lessor degree
output_dim=156, hiddenState=(78x1), Dropout(0.45): around 0.735 2:1   -overfitting lessor degree
output_dim=78, hiddenState=(49x1), Dropout(0.45): around 0.     2:1   -overfitting lessor degree
'''