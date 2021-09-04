# Coping with missing values
# 1. delete the rows with missing values
# 2. substitute the missing values with an arbitrary number; usually zeros
# 3. substitute the missing values with neighbors' values
# 4. substitute the missing values with the median value
# 5. intepolation: various ways to estimate what is missing
#                  here a linear esimation was used
# 6. use boost type models which are not hugely affected by presence of missing values


import numpy as np
import pandas as pd
from datetime import datetime

from pandas.core.series import Series

data_date = ['08/13/2021', '08/14/2021', '08/15/2021', '08/16/2021', '08/17/2021']
dates = pd.to_datetime(data_date)

missing_values = Series([1, np.nan, np.nan, 8, 10], index=dates)
interpolated = missing_values.interpolate()

print('from:', missing_values)
print('interpolated:', interpolated)

'''
from:                   interpolated: 
2021-08-13     1.0      2021-08-13     1.000000
2021-08-14     NaN      2021-08-14     3.333333
2021-08-15     NaN      2021-08-15     5.666667
2021-08-16     8.0      2021-08-16     8.000000
2021-08-17    10.0      2021-08-17    10.000000
'''