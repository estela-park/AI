import pandas as pd
import os
# print(os.getcwd()): D:\study
import chardet
# rawdata = open('./Dacon/_data/train_data.csv', 'rb').read()
# result = chardet.detect(rawdata)
# charenc = result['encoding']: utf-8
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

train = pd.read_csv('./Dacon/_data/train_data.csv', encoding='utf-8')
# print(train.keys()): ['index', 'title', 'topic_idx']
train = train.set_index('index')

test = pd.read_csv('./Dacon/_data/test_data.csv', encoding='utf-8')
# print(test.keys()): ['index', 'title']
test = test.set_index('index')

train_x = train['title']
train_y = train['topic_idx']
test_x = test['title']

text = ' '.join(train_x)
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
# 101081 <class 'dict'>

train_x = tokenizer.texts_to_sequences(train_x)
test_x = tokenizer.texts_to_sequences(test_x)

# maximum lenth of articles: 13
# verage length of articles: 6.7
train_x = pad_sequences(train_x, maxlen=7, padding='post')

# len(train_y.unique()): 7

train_y = to_categorical(train_y)
# (45654, 7)