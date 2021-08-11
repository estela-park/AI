import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils import all_estimators


# 1. Data-prep
dataset = load_boston()

x = dataset.data   
y = dataset.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85) 
kfold = KFold(n_splits=5, shuffle=True, random_state=99)


# 2, 3, 4. Modelling, Training, Evaluation
all_model = all_estimators('regressor')
count = 0
for name, model in all_model:
    try: 
        model = model()
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print(f'with {name}, ', end='')
        print('k-value: 5, accuracy:', scores, round(np.mean(scores), 4))

        count += 1

    except:
        print(f'{name} will not work for this particular data-set')
        continue

print('the number of working models:', count)
print('the number of candidate models:', len(all_model))

'''
with ARDRegression,                 k-value: 5, accuracy: [0.7151990 0.68185734 0.74073521 0.78206685 0.59901128] 0.7038
with AdaBoostRegressor,             k-value: 5, accuracy: [0.8249527 0.84524673 0.86404777 0.83736994 0.73254205] 0.8208
with BaggingRegressor,              k-value: 5, accuracy: [0.8964793 0.79376846 0.833866   0.90193094 0.68082711] 0.8214
with BayesianRidge,                 k-value: 5, accuracy: [0.6883436 0.66850771 0.74754935 0.77701721 0.59775809] 0.6958
with CCA,                           k-value: 5, accuracy: [0.6125251 0.64317702 0.7354062  0.80540959 0.47441667] 0.6542
with DecisionTreeRegressor,         k-value: 5, accuracy: [0.7290592 0.69946413 0.82285423 0.87178294 0.63823257] 0.7523
with DummyRegressor,                k-value: 5, accuracy: [-0.03673 -0.00527359 -0.000541 -0.0136994 -0.00051503] -0.0114
with ElasticNet,                    k-value: 5, accuracy: [0.6601031 0.67095511 0.69423136 0.71063157 0.55321146] 0.6578
with ElasticNetCV,                  k-value: 5, accuracy: [0.6467671 0.66281198 0.68406403 0.69718106 0.52801856] 0.6438
with ExtraTreeRegressor,            k-value: 5, accuracy: [ 0.527366 0.70160676 0.5294814  0.77155795 -0.0769855] 0.4906
with ExtraTreesRegressor,           k-value: 5, accuracy: [0.8815580 0.86189553 0.91680044 0.87778975 0.84421435] 0.8765
with GammaRegressor,                k-value: 5, accuracy: [-0.03713 -0.0055281 -0.0005597 -0.0138762 -0.0005589 ] -0.0115
with GaussianProcessRegressor,      k-value: 5, accuracy: [-6.28301 -5.1335677 -5.2905746 -4.6396032 -8.8865682 ] -6.0467
with GradientBoostingRegressor,     k-value: 5, accuracy: [0.9035811 0.89093561 0.88604443 0.91267118 0.78674242] 0.876
with HistGradientBoostingRegressor, k-value: 5, accuracy: [0.8838483 0.83196563 0.87261963 0.86799137 0.75992187] 0.8433
with HuberRegressor,                k-value: 5, accuracy: [0.6574286 0.50720958 0.74956618 0.68818673 0.442963  ] 0.6091
with IsotonicRegression,            k-value: 5, accuracy: [nan       nan        nan        nan        nan       ] nan
with KNeighborsRegressor,           k-value: 5, accuracy: [0.4658298 0.49288362 0.57302855 0.51685823 0.29584522] 0.4689
with KernelRidge,                   k-value: 5, accuracy: [0.6645940 0.63061062 0.77590456 0.78531182 0.53328125] 0.6779
with Lars,                          k-value: 5, accuracy: [0.7083394 0.67530827 0.74945245 0.78916974 0.6048925 ] 0.7054
with LarsCV,                        k-value: 5, accuracy: [0.7138504 0.67927835 0.74962907 0.78815885 0.60848733] 0.7079
with Lasso,                         k-value: 5, accuracy: [0.6622546 0.66427576 0.67509637 0.69205938 0.54937222] 0.6486
with LassoCV,                       k-value: 5, accuracy: [0.6758295 0.66541829 0.70897824 0.73385401 0.56705481] 0.6702
with LassoLars,                     k-value: 5, accuracy: [-0.03676 -0.0052735 -0.0005430 -0.0136994 -0.00051503] -0.0114
with LassoLarsCV,                   k-value: 5, accuracy: [0.7103187 0.68372578 0.74962907 0.78916974 0.6048925 ] 0.7075
with LassoLarsIC,                   k-value: 5, accuracy: [0.716975  0.68233979 0.75400536 0.78374747 0.60012151] 0.7074
with LinearRegression,              k-value: 5, accuracy: [0.7083394 0.68372578 0.74945245 0.78916974 0.6048925 ] 0.7071
with LinearSVR,                     k-value: 5, accuracy: [0.1411719 0.5270083  0.52603942 0.68818582 0.47185192] 0.4709
with MLPRegressor,                  k-value: 5, accuracy: [0.598202  0.66234048 0.686593  -0.325378   0.34839734] 0.394
with MultiTaskElasticNet,           k-value: 5, accuracy: [nan       nan        nan        nan        nan       ] nan
with MultiTaskElasticNetCV,         k-value: 5, accuracy: [nan       nan        nan        nan        nan       ] nan
with MultiTaskLasso,                k-value: 5, accuracy: [nan       nan        nan        nan        nan       ] nan
with MultiTaskLassoCV,              k-value: 5, accuracy: [nan       nan        nan        nan        nan       ] nan
with NuSVR,                         k-value: 5, accuracy: [0.1655560 0.14157658 0.30205982 0.23406147 0.1521796 ] 0.1991
with OrthogonalMatchingPursuit,     k-value: 5, accuracy: [0.4998897 0.5495736  0.57466445 0.57784813 0.42060558] 0.5245
with OrthogonalMatchingPursuitCV,   k-value: 5, accuracy: [0.6788451 0.64689454 0.74574613 0.74405484 0.53787148] 0.6707
with PLSCanonical,                  k-value: 5, accuracy: [-3.08229 -1.9195530 -2.0917064 -0.9954997 -3.65019016] -2.3478
with PLSRegression,                 k-value: 5, accuracy: [0.6765218 0.64097603 0.77421868 0.75485506 0.55259117] 0.6798
with PassiveAggressiveRegressor,    k-value: 5, accuracy: [0.1311932 0.1418786 -0.16331553 0.2376896 -2.64714153] -0.4599       
with PoissonRegressor,              k-value: 5, accuracy: [0.7296465 0.73797907 0.7803662  0.85404372 0.57674199] 0.7358
with RANSACRegressor,               k-value: 5, accuracy: [0.2946613 0.25585635 0.42730635 0.7063123  0.58046709] 0.4529
with RadiusNeighborsRegressor,      k-value: 5, accuracy: [nan       nan        nan        nan        nan       ] nan
with RandomForestRegressor,         k-value: 5, accuracy: [0.8810929 0.84433169 0.88190934 0.88926065 0.77539204] 0.8544
with Ridge,                         k-value: 5, accuracy: [0.7009919 0.67630879 0.75783778 0.78900228 0.60415712] 0.7057
with RidgeCV,                       k-value: 5, accuracy: [0.7078619 0.68269661 0.75161135 0.78946884 0.60541143] 0.7074
with SGDRegressor,                  k-value: 5, accuracy: [-1.3e+26 -3.53e+26  -4.727e+26 -2.252e+26 -4.5478e+26] -3.283922877790981e+26
with SVR,                           k-value: 5, accuracy: [0.1475955 0.10643152 0.27274061 0.20362708 0.12591499] 0.1713
with TheilSenRegressor,             k-value: 5, accuracy: [0.6440280 0.60074264 0.79513118 0.76123279 0.61235274] 0.6827
with TransformedTargetRegressor,    k-value: 5, accuracy: [0.7083394 0.68372578 0.74945245 0.78916974 0.6048925 ] 0.7071
with TweedieRegressor,              k-value: 5, accuracy: [0.6535307 0.6548311  0.68858529 0.69200395 0.50865945] 0.6395

MultiOutputRegressor will not work for this particular data-set
RegressorChain will not work for this particular data-set
StackingRegressor will not work for this particular data-set
VotingRegressor will not work for this particular data-set

the number of working models: 50
the number of candidate models: 54
'''