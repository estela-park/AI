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


data = pd.read_csv('../_data/_solo/pusan_vax.txt', sep=',')
temp = data.keys()[2:]
data = pd.DataFrame(temp, index=index_PC)
print(data)
data.to_csv('../_save/_solo/vax.csv', encoding='EUC-KR')