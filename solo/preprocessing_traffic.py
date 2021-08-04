import chardet
import pandas as pd


# rawdata = open('../_data/_solo/TCS_35_04_01_780205.csv', 'rb').read()
# result = chardet.detect(rawdata)
#charenc = result['encoding']: EUC-KR


dataset_april = pd.read_csv('../_data/_solo/april.csv', encoding='EUC-KR')
dataset_may = pd.read_csv('../_data/_solo/may.csv', encoding='EUC-KR')
dataset_june = pd.read_csv('../_data/_solo/june.csv', encoding='EUC-KR')
dataset_july = pd.read_csv('../_data/_solo/july.csv', encoding='EUC-KR')
# dataset.keys(): 집계일자, 출발영업소코드, 도착영업소코드, 출발영업소명, 도착영업소명, 도착지방향1종교통량, 도착지방향2종교통량,
#                 도착지방향3종교통량, 도착지방향4종교통량, 도착지방향5종교통량, 도착지방향6종교통량, 도착지방향총교통량, 
#                 출발지방향1종교통량, 출발지방향2종교통량, 출발지방향3종교통량, 출발지방향4종교통량, 출발지방향5종교통량, 
#                 출발지방향6종교통량, 출발지방향총교통량
#                 101: 서울, 140: 부산
# dataset['도착영업소명'].unique(): ['서울' ... '부산' ... ]

data_april = dataset_april[dataset_april['도착영업소명'] == '부산']
data_april = data_april[data_april['출발영업소명'] == '서울']
data_may = dataset_may[dataset_may['도착영업소명'] == '부산']
data_may = data_may[data_may['출발영업소명'] == '서울']
data_june = dataset_june[dataset_june['도착영업소명'] == '부산']
data_june = data_june[data_june['출발영업소명'] == '서울']
data_july = dataset_july[dataset_july['도착영업소명'] == '부산']
data_july = data_july[data_july['출발영업소명'] == '서울']

data_april = data_april.set_index('집계일자')['도착지방향총교통량']
data_may = data_may.set_index('집계일자')['도착지방향총교통량']
data_june = data_june.set_index('집계일자')['도착지방향총교통량']
data_july = data_july.set_index('집계일자')['도착지방향총교통량']

data = pd.concat([data_april, data_may, data_june, data_july], axis=0)
print(data)

data.to_csv('../_save/_solo/traffic.csv', encoding='EUC-KR')