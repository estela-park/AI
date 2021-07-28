from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Bidirectional
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

inputL = Input((5,))
hl = Embedding(input_dim=28, output_dim=11)(inputL)
hl = Bidirectional(LSTM(32))(hl)
outputL = Dense(1, activation='sigmoid')(hl)

model = Model(inputs=inputL, outputs=outputL)

model.summary()

start = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=120, batch_size=16)
end = time.time() - start

acc = model.evaluate(pad_x, labels)[1]
print('acc:', acc)
print('time taken:',end)

'''
Model: "model"
acc: 1.0
time taken: 4 seconds
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 5)]               0
_________________________________________________________________
embedding (Embedding)        (None, 5, 11)             308
_________________________________________________________________
bidirectional (Bidirectional (None, 64)                11264
_________________________________________________________________
dense (Dense)                (None, 1)                 65
=================================================================
Total params: 11,637
'''