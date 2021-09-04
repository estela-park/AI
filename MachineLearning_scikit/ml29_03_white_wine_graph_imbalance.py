# this dataset is very imbalanced

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../_data/white_wine.csv', sep=';', index_col=None, header=0)

# groupby and count are pandas's methods
count_data = data.groupby('quality')['quality'].count()
# type(data.groupby('quality')['quality']): <class 'pandas.core.groupby.generic.SeriesGroupBy'>
# type(count_data): <class 'pandas.core.series.Series'>

plt.bar(count_data.index, count_data)
plt.show()

'''
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
'''