import time
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


# 1. Data-prep
datasets = load_boston()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=78)

scaler_mm = MinMaxScaler()
x_train_mm = scaler_mm.fit_transform(x_train)
x_test_mm = scaler_mm.transform(x_test)

# 2. Modelling
model = XGBRegressor(n_estimators=240, learning_rate=0.01, n_jobs=8,
                     tree_method='gpu_hist', predictor='gpu_predictor', # 'cpu_predictor'
                     gpu_id=1)

# eventhough the model is trained on GPU, 
# without specification predictions are performed using the CPU.
#
# gpu_exact vs gpu_hist
#  > gpu_exact	The standard XGBoost tree construction algorithm. 
#               Performs exact search for splits. 
#               Slower and uses considerably more memory than gpu_hist.
#  > gpu_hist	Equivalent to the XGBoost fast histogram algorithm. 
#               Much faster and uses considerably less memory.
# Multiple GPUs can be used with the gpu_hist tree method using the n_gpus parameter. 
# which defaults to 1. 
# If this is set to -1 all available GPUs will be used.
# As with GPU vs. CPU, multi-GPU is not gueranteed to be faster than a single GPU, 
#  due to PCI bus bandwidth that can limit performance.


# Training & Evaluation
start = time.time()
model.fit(x_train_mm, y_train, verbose=1, eval_metric='rmse', eval_set=[(x_test_mm, y_test)], early_stopping_rounds=8)
end = time.time() - start

score = model.score(x_test_mm, y_test)
predict = model.predict(x_test_mm)
r2 = r2_score(y_test, predict)
print('it took', end//60, 'minutes and', end%60,'seconds')
print('model.score:', score, '& R2score:', r2)

# it took 3 seconds
# model.score: 0.81 & R2score: 0.81