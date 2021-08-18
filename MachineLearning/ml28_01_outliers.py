# When to deal with outliers with quartile ranging, 
# data may not always conform to normal distribution.
#
# When the data has multiple peaks or is skewed greatly,
# deleting outliers with standardization distorts the data.
# 
# Normal quartile range: quadruple of Interquartile Range
#
# Options
#  > 1. Deletion
#  > 2. Nan -> bogan?// linear
#  > 3. Similar to missing value treatment
#  > 4. Scaler: Robust, Quantile
#  > 5. Model: Tree-likes, XG, LGBM


import numpy as np


# it returns the indexes of outliers
def outliers(data_out):
    lower_quartile, mid, upper_quartile = np.percentile(data_out, [25, 50, 75])
    iqr = upper_quartile - lower_quartile
    print('***********Quartile Index***********')
    print('lowest:', np.percentile(data_out, 0))
    print('Q1:', lower_quartile)
    print('Q2:', mid)
    print('Q3:', upper_quartile)
    print('highest:', np.percentile(data_out, 100))
    lower_bound = lower_quartile - iqr*1.5
    upper_bound = upper_quartile + iqr*1.5
    return np.where((data_out > upper_bound) | (data_out < lower_bound))


aaa = np.array([1, 2, -1000, 4, 5, 6, 7, 8, 90, 100, 500])
outlier_loc = outliers(aaa)
print('Indexes of outliers:', outlier_loc)