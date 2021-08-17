parameters = [
    {'n_estimators': [90, 100], 'learning_rate': [0.1, 0.001, 0.5], 'max_depth': [4, 5, 6],
     'colsample_bytree': [0.6, 0.9, 1], 'colsample_bylevel': [0.6, 0.7, 0.9], 'tree_method': ['gpu_hist']}
]

import pandas as pd
import numpy as np
import re
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# start = time.time()
# train = pd.read_csv('../_data/dacon/train.csv', encoding='utf-8')
# 7.0 seconds to load
# end = time.time() - start
# train = train.set_index('index')
# print(train)
# print(end//1,'seconds to load')
# np.save('./tech/_data/train.npy', arr=train)

# start = time.time()
# test = pd.read_csv('../_data/dacon/test.csv', encoding='utf-8')
# 1.0 seconds to load
# end = time.time() - start
# test = test.set_index('index')
# print(test)
# print(end//1,'seconds to load')
# np.save('./tech/_data/test.npy', arr=test)

train = np.load('../_data/_tech/train.npy', allow_pickle=True)
train = pd.DataFrame(train)

train_x = train.iloc[:, [2, 9, 10]]
# (174304, 3)
train_y = train.iloc[:, 11]
# (174304,)

# train_x.isnull().sum(): 2: 0, 9: 3028, 10: 3087
train_x = train_x.fillna(value={9:'널값', 10:'Null'})
train_x.rename(columns = {2: 'department', 9: 'key_ko', 10: 'key_en'}, inplace = True)

arr_ko = np.array(train_x['key_ko'])
dictionary_ko = []
dict_length_ko = []
# avg: 5, max: 38 -> 10
cnt = 0
for line in arr_ko:
    dictionary_ko.append([x.strip() for x in line.split(',')])
    dict_length_ko.append(len(dictionary_ko[cnt]))
    cnt += 1

tokenizer_ko = Tokenizer()
text_ko = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", ''.join(map(' '.join, dictionary_ko)))
tokenizer_ko.fit_on_texts([text_ko])
# 269796

arr_en = np.array(train_x['key_en'])
dictionary_en = []
dict_length_en = []
# avg: 5, max: 22 -> 10
cnt = 0
for line in arr_en:
    dictionary_en.append([x.strip() for x in line.split(',')])
    dict_length_en.append(len(dictionary_en[cnt]))
    cnt += 1

tokenizer_en = Tokenizer()
tokenizer_en.fit_on_texts(dictionary_en)
# 273031


train_x['key_ko'] = tokenizer_ko.texts_to_sequences(train_x['key_ko'])

train_x['key_ko'] = pad_sequences(train_x['key_ko'], maxlen=10, padding='post')