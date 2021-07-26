import pandas as pd

temp = pd.read_csv('./Dacon/_save/pre_submission.csv', encoding='utf-8')

temp.rename( columns={'Unnamed: 0':'index'}, inplace=True )
temp.rename( columns={'0':'topic_idx'}, inplace=True )
temp['index'] = temp['index'] + 45654

temp = temp.set_index('index')

temp.to_csv('./Dacon/_save/submission.csv')