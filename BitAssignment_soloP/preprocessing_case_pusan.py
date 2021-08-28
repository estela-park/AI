import pandas as pd

# data = pd.read_csv('../_save/_solo/traffic.csv', encoding='EUC-KR')
# print(type(data.iloc[3, 0])): int64


index_PC = []

for i in range(1, 31):
    index_PC.append(20210400 + i)
for i in range(1, 32):
    index_PC.append(20210500 + i)
for i in range(1, 31):
    index_PC.append(20210600 + i)
for i in range(1, 32):
    index_PC.append(20210700 + i)
for i in range(1, 4):
    index_PC.append(20210800 + i)


data = pd.read_csv('../_data/_solo/pusan_case.csv')
temp = data.keys()[2:]
data = pd.DataFrame(temp, index=index_PC)

data.to_csv('../_save/_solo/cases_pusan.csv', encoding='EUC-KR')