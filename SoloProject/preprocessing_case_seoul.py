import pandas as pd


index_PC = []

for i in range(17, 31):
    index_PC.append(20210400 + i)
for i in range(1, 32):
    index_PC.append(20210500 + i)
for i in range(1, 31):
    index_PC.append(20210600 + i)
for i in range(1, 32):
    index_PC.append(20210700 + i)
for i in range(1, 4):
    index_PC.append(20210800 + i)


data = pd.read_csv('../_data/_solo/seoul_case.txt', sep=',')
temp = data.keys()[2:]
# each key, column should hold unique value >> 200 200.1 200.2
data = pd.DataFrame(temp, index=index_PC)

data.to_csv('../_save/_solo/cases_seoul.csv', encoding='EUC-KR')