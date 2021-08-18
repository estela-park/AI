import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


datasets = load_iris()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)

model = KMeans(len(np.unique(datasets.target)), max_iter=300, random_state=77)
model.fit(df)

df['cluster'] = model.labels_
df['target'] = datasets.target
iris_result = df.groupby(['target', 'cluster'])['sepal length (cm)'].count()
print(iris_result)


'''
****Cluster****
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
 1 1 1 1 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 2 0 2 2 2 2 0 2 2 2 2 2 2 0 0 2 2 2 2 0 2 0 2 0 2 2 0 0 2 2 2 2 2 0 2 2 2 2 
 0 2 2 2 0 2 2 2 0 2 2 0]
*****Label*****
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 
 2 2 2 2 2 2 2 2 2 2 2 2]

target  cluster
0       1          50
1       0          48
        2           2 - missed values
2       0          14 - missed values
        2          36
'''