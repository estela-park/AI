import matplotlib.pyplot as plt
from xgboost import XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 1. Data-prep
dataset = load_boston()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=76)


# 2. Modelling
model = XGBRegressor()


# 3. Training
model.fit(x_train, y_train)


# 4. Evaluation
score = model.score(x_test, y_test)
print('model.score:', score)
print(model.feature_importances_)

# the tree in model arg must be Booster, XGBModel or dict instance
plot_importance(model)
plt.show()