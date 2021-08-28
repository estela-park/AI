# PCA: combines independent variables resulting in elimination of correlation btn variables and dimensionality reduction
# Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.
# PCA doesn't do the scaling, a scaler should be used separately.

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA


# 1. Data-prep
datasets = load_boston()
x = datasets.data
# (506, 13)
y = datasets.target
# (506,)

# Number of components to keep. if n_components is not set all components are kept
# n_components=#
pca = PCA(n_components=7)
x = pca.fit_transform(x)
# x.shape: (442, n_components)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=78)


# 2. Modelling
model = DecisionTreeRegressor()


# 3. Training
model.fit(x_train, y_train)


# 4. Evaluation
score = model.score(x_test, y_test)
print('model.score:', score)
print(model.feature_importances_)

'''
-- Without PCA
model.score: 0.7841012245764913
[0.03360512 0.00127667 0.0115701  0.00114898 0.02404706 0.25112467
 0.01123617 0.08753423 0.00308223 0.01339983 0.0079122  0.00717746
 0.54688526]

-- With PCA
- n_components=10
model.score: 0.4582858326455479
[0.26128165 0.03449115 0.09704542 0.04925991 0.00951371 0.33073466
 0.03642797 0.04755134 0.02300534 0.11068884]
- n_components=7
model.score: 0.5261252714007512
[0.30143503 0.04011575 0.1867181  0.03571344 0.03743276 0.34432365
 0.05426127]
'''