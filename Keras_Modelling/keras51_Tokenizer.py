# tokenizer: map the word to a integer, the more frequent word first

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


text = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다'

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
# token.word_index: {'진짜': 1, '마구': 2, '나는': 3, '매우': 4, '맛있는': 5, '밥을': 6, '먹었다': 7}

x = tokenizer.texts_to_sequences([text])
# [[3, 1, 4, 5, 6, 1, 2, 2, 7]]
x = to_categorical(x)
# [[[0. 0. 0. 1. 0. 0. 0. 0.]
# [0. 1. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 1. 0. 0. 0.]
# [0. 0. 0. 0. 0. 1. 0. 0.]
# [0. 0. 0. 0. 0. 0. 1. 0.]
# [0. 1. 0. 0. 0. 0. 0. 0.]
# [0. 0. 1. 0. 0. 0. 0. 0.]
# [0. 0. 1. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 1.]]]
