import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# y = 2x + 3
x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-3, 65, -19, 11, 3, 47, -1, -7, -47, -25]

# this shows as a clean line
plt.plot(x, y)
plt.show()

df = pd.DataFrame({'X':x, 'Y':y})
x_train = df.loc[:, 'X'].values.reshape(-1, 1)
y_train = df.loc[:, 'Y']

model = LinearRegression()
model.fit(x_train, y_train)
print('model.score:', model.score(x_train, y_train), '& slope:', model.coef_, '& intercept:', model.intercept_)
# model.score: 1.0 & slope: [2.] & intercept: 3.0