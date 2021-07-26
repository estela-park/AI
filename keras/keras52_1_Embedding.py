from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
import numpy as np

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다.', 
        '한 번 더 보고 싶네요', '글쎄요', '별로에요', '생각보다 지루해요', '연기가 어색해요', 
        '재미없어요', '너무 재미없다', '참 재밌네요', '청순이가 잘 생기긴 했어요']

labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs)
# {'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고에요': 5, '만든': 6, '영화에요': 7, 
# '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, 
# '싶네요': 15, '글쎄요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, 
# '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '청순이가': 25, 
# '생기긴': 26, '했어요': 27}

x = tokenizer.texts_to_sequences(docs)
# [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], 
# [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]
# pad_x = pad_sequences(x, maxlen=max(map(lambda x: len(x), x)), padding='pre')
pad_x = pad_sequences(x, maxlen=5, padding='post')
# [[ 0  0  0  2  4]
#  [ 0  0  0  1  5]
#  [ 0  1  3  6  7]
#  [ 0  0  8  9 10]
#  [11 12 13 14 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0  0 17]
#  [ 0  0  0 18 19]
#  [ 0  0  0 20 21]
#  [ 0  0  0  0 22]
#  [ 0  0  0  2 23]
#  [ 0  0  0  1 24]
#  [ 0 25  3 26 27]]

model = Sequential()
model.add(Embedding(input_dim=28, output_dim=11, input_length=5))
# Embedding: encodes words to one-hot vector then map the vectors to 2D coordinate system.
# input_dim: word_index+1 - 단어의 종류
# input_length: 문장의 단어 개수
# output_dim: arbitrary
# if given as positional, input_dim, output_dim, input_length
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
# 1: positive, 0: negative
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 11)             308
_________________________________________________________________
lstm (LSTM)                  (None, 32)                5632
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
the number of parameters for Embedding layer:
 > input_dim(the number of words)*output_dim(the number of nodes)
 > Embedding layers output when args are given as (input_dim, output_dim), positional
   > Output Shape: (None=batch_size, None=the number of sentence, output_dim=nodes)
'''

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=120, batch_size=16)

acc = model.evaluate(pad_x, labels)[1]
print('acc:', acc)