import time
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

dataset = load_diabetes()

x = dataset.data   
print(x.shape)
# (506, 13)
y = dataset.target
print(y.shape)
# (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85)

max_abs_scaler = MaxAbsScaler()
max_abs_scaler.fit(x_train)
x_train_ma = max_abs_scaler.transform(x_train)
x_test_ma = max_abs_scaler.transform(x_test)

start = time.time()

model_ma = KNeighborsRegressor()

# except data, most parameters are set default
model_ma.fit(x_train_ma, y_train)

# evaluate => score, it gives back accuracy.
result_ma = model_ma.score(x_test_ma, y_test)
predict_ma = model_ma.predict(x_test_ma)

end = time.time() - start

# Accuracy computing
acc = r2_score(y_test, predict_ma)

print('it took',end)
print('accuracy score:', acc, end='')
print(', prediction for ',y_test[:8],'is',predict_ma[:8])

'''
**KNeighborsRegressor
    accuracy score: 0.5629675570380624
    prediction for  [178. 275. 270. 147. 78. 160. 39. 233.] is [165.4 193.2 263.4 134.8 89.8 149.2 79.4 218.2]
**LinearRegression
    accuracy score: 0.38326359915443564
    prediction for  [166. 170. 74. 96. 242. 52. 259. 54.] is [207.36 135.34 88.18 87.09 168.38 176.11 239.15 100.97]
**DecisionTreeRegressor
    accuracy score: -0.4363844709158009
    prediction for  [185. 209. 263. 185.  59.  90.  53. 237.] is [ 90.  69. 288. 189.  71. 116.  64. 288.]
    accuracy score: -0.05270184392993116
    prediction for  [ 48. 220. 245.  49. 110.  90. 261.  60.] is [191. 303. 277. 179. 197. 200. 274. 121.]
    accuracy score: -0.14647546949438817
    prediction for  [214. 279. 177. 132. 137.  68. 142.  96.] is [ 83. 306.  81.  50.  51.  49.  55.  48.]
**RandomForestRegressor
    accuracy score: 0.38315586869720486
    prediction for  [151. 97. 110. 53. 121. 201. 178. 94.] is [176.46 107.92 140.29  81.54 152.7  107.41  91.85 159.21]
'''