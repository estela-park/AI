# fn that returns indexes and the number of outliers for every feature
import numpy as np


def outliers_by_feature(data_out):
    outliers = {}
    for i in range(data_out.shape[1]):
        lower_quartile, mid, upper_quartile = np.percentile(data_out[:, i], [25, 50, 75])
        iqr = upper_quartile - lower_quartile
        print('***********Quartile Index***********')
        print('lowest:', np.percentile(data_out[:, i], 0))
        print('Q1:', lower_quartile)
        print('Q2:', mid)
        print('Q3:', upper_quartile)
        print('highest:', np.percentile(data_out[:, i], 100))
        lower_bound = lower_quartile - iqr*1.5
        upper_bound = upper_quartile + iqr*1.5
        outliers[i] = (np.where((data_out[:, i] > upper_bound) | (data_out[:, i] < lower_bound)), sum(1 for x in data_out[:, i] if (x > upper_bound) | (x < lower_bound)))
    return  outliers


aaa = np.array([[1, 2, 10000, 3, 4, 6, 7, 8, 90, 100],
                [1000, 2000, 3, 4000, 5000, 6000, 7000, 8, 9000, 10000]]).transpose()
# (10, features=2)

# dict > 'column num': ([index], # of outliers)
print(outliers_by_feature(aaa))

'''
***********Quartile Index***********
lowest: 1.0
Q1: 3.25
Q2: 6.5
Q3: 69.5
highest: 10000.0
***********Quartile Index***********
lowest: 3.0
Q1: 1250.0
Q2: 4500.0
Q3: 6750.0
highest: 10000.0

{0: ((array([2], dtype=int64),), 1), 1: ((array([], dtype=int64),), 0)}
'''