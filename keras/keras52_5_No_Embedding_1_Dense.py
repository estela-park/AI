from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import time

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다.', 
        '한 번 더 보고 싶네요', '글쎄요', '별로에요', '생각보다 지루해요', '연기가 어색해요', 
        '재미없어요', '너무 재미없다', '참 재밌네요', '청순이가 잘 생기긴 했어요']

labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs)

x = tokenizer.texts_to_sequences(docs)
pad_x = pad_sequences(x, maxlen=5, padding='post')

model = Sequential()
model.add(Dense(32, input_shape=(5,)))
model.add(Dense(32))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

start = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=120, batch_size=16)
end = time.time() - start

acc = model.evaluate(pad_x, labels)[1]
print('time:',end,'seconds')
print('acc:', acc)

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 32)                192
_________________________________________________________________
dense_1 (Dense)              (None, 32)                1056
_________________________________________________________________
flatten (Flatten)            (None, 32)                0
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33
=================================================================
Total params: 1,281
time: 1 second
acc: 0.9230769276618958
'''