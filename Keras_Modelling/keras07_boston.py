from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

datasets = load_boston()
x = datasets.data
y = datasets.target

# x.shape: (506, 13)
# y.shape: (506,)

# datasets.feature_names: ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
# datasets.DESCR: 
# This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
# 
# The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
# prices and the demand for clean air', J. Environ. Economics & Management,
#vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
# ...', Wiley, 1980.   N.B. Various transformations are used in the table on
# pages 244-261 of the latter.
# 
# The Boston house-price data has been used in many machine learning papers that address regression
# problems.
#
# .. topic:: References
# 
#   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.       
#   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
# Boston house prices dataset
# ---------------------------
#
# **Data Set Characteristics:**
#
#     :Number of Instances: 506
#
#     :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
#
#     :Attribute Information (in order):
#         - CRIM     per capita crime rate by town
#         - ZN       proportion of residential land zoned for lots over 
# 25,000 sq.ft.
#        - INDUS    proportion of non-retail business acres per town   
#        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#        - NOX      nitric oxides concentration (parts per 10 million) 
#        - RM       average number of rooms per dwelling
#        - AGE      proportion of owner-occupied units built prior to 1940
#        - DIS      weighted distances to five Boston employment centres
#        - RAD      index of accessibility to radial highways
#        - TAX      full-value property-tax rate per $10,000
#        - PTRATIO  pupil-teacher ratio by town
#        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#        - LSTAT    % lower status of the population
#        - MEDV     Median value of owner-occupied homes in $1000's    
#
#    :Missing Attribute Values: None
#
#    :Creator: Harrison, D. and Rubinfeld, D.L.
#
# This is a copy of UCI ML housing dataset.
# https://archive.ics.uci.edu/ml/machine-learning-databases/housing/    

# type(x): ndarray
# type(y): ndarray

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=354, random_state=47)

model = Sequential()
model.add(Dense(26, input_dim=13))
model.add(Dense(52))
model.add(Dense(13))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# bigger epoch size doesn't make whole lot of difference
model.fit(x_train, y_train, epochs=800, batch_size=1)

loss = model.evaluate(x_test, y_test)
y_predicted = model.predict(x_test)
print(type(y_test))
print(type(y_predicted))
r2 = r2_score(y_test, y_predicted)

print('loss:', loss,'accuracy:',r2)