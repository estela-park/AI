import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils import all_estimators


# 1. Data-prep
dataset = load_diabetes()

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
with ARDRegression,                 k-value: 5, accuracy: [0.4340566 0.4193550 0.57011376 0.56728747 0.50526611] 0.4992
with AdaBoostRegressor,             k-value: 5, accuracy: [0.4170036 0.3426474 0.55916081 0.52222571 0.46776182] 0.4618
with BaggingRegressor,              k-value: 5, accuracy: [0.3012658 0.3326059 0.44859974 0.50291768 0.42428563] 0.4019
with BayesianRidge,                 k-value: 5, accuracy: [0.4322357 0.4218626 0.58984893 0.58423732 0.51557058] 0.5088
with CCA,                           k-value: 5, accuracy: [0.2721500 0.1878721 0.23191421 0.42534237 0.22014146] 0.2675
with DecisionTreeRegressor,         k-value: 5, accuracy: [-0.121437 0.084480 -0.22784155 0.1863847 -0.02724669] -0.0211
with DummyRegressor,                k-value: 5, accuracy: [-3.1e-05 -8.59e-03 -8.160e-03 -2.230e-02 -9.9281e-06] -0.0078
with ElasticNet,                    k-value: 5, accuracy: [0.0088986 0.0003908 0.00271964 -0.013011  0.0099358 ] 0.0018
with ElasticNetCV,                  k-value: 5, accuracy: [0.4048659 0.3812538 0.50230225 0.5235363  0.47477029] 0.4573
with ExtraTreeRegressor,            k-value: 5, accuracy: [-0.272856 0.267888 -0.28693268 0.29318778 0.13170117] 0.0266
with ExtraTreesRegressor,           k-value: 5, accuracy: [0.3904347 0.3773126 0.53128315 0.52401577 0.45462424] 0.4555
with GammaRegressor,                k-value: 5, accuracy: [0.006194 -0.001548 -0.0002754 -0.01548991 0.00634889] -0.001
with GaussianProcessRegressor,      k-value: 5, accuracy: [-9.80495 -16.38743 -11.770480 -16.779345 -6.62684802] -12.2738
with GradientBoostingRegressor,     k-value: 5, accuracy: [0.3560015 0.3494856 0.43642806 0.56481817 0.49660277] 0.4407
with HistGradientBoostingRegressor, k-value: 5, accuracy: [0.3237060 0.296731  0.46278708 0.46987951 0.4145346 ] 0.3935
with HuberRegressor,                k-value: 5, accuracy: [0.4223357 0.4241505 0.59696249 0.56741063 0.49752821] 0.5017
with IsotonicRegression,            k-value: 5, accuracy: [nan       nan       nan        nan        nan       ] nan
with KNeighborsRegressor,           k-value: 5, accuracy: [0.4297643 0.3243294 0.43723522 0.45542472 0.38260685] 0.4059
with KernelRidge,                   k-value: 5, accuracy: [-3.40781 -3.639807 -3.9780787 -3.2155919 -3.94305453] -3.6369
with Lars,                          k-value: 5, accuracy: [0.140403  0.417815 -3.15560945 0.5851182  0.51332843] -0.2998
with LarsCV,                        k-value: 5, accuracy: [0.442296  0.4196188 0.54648119 0.56821178 0.51558248] 0.4984
with Lasso,                         k-value: 5, accuracy: [0.3536548 0.3391594 0.42469209 0.34957168 0.36897706] 0.3672
with LassoCV,                       k-value: 5, accuracy: [0.4327002 0.4197121 0.57328852 0.57132851 0.51199082] 0.5018
with LassoLars,                     k-value: 5, accuracy: [0.3834489 0.3637046 0.46858939 0.40543905 0.40805008] 0.4058
with LassoLarsCV,                   k-value: 5, accuracy: [0.4333312 0.4196188 0.58763707 0.56821178 0.51063499] 0.5039
with LassoLarsIC,                   k-value: 5, accuracy: [0.4303597 0.4199000 0.59103701 0.57525801 0.51078091] 0.5055
with LinearRegression,              k-value: 5, accuracy: [0.4251577 0.4219485 0.59166041 0.58511829 0.51332843] 0.5074
with LinearSVR,                     k-value: 5, accuracy: [-0.45734 -0.330005 -0.4237200 -0.5018290 -0.49225732] -0.441
with MLPRegressor,                  k-value: 5, accuracy: [-2.91835 -2.781832 -3.4483123 -2.8190560 -3.36728847] -3.067
with MultiTaskElasticNet,           k-value: 5, accuracy: [nan       nan       nan        nan        nan       ] nan
with MultiTaskElasticNetCV,         k-value: 5, accuracy: [nan       nan       nan        nan        nan       ] nan
with MultiTaskLasso,                k-value: 5, accuracy: [nan       nan       nan        nan        nan       ] nan
with MultiTaskLassoCV,              k-value: 5, accuracy: [nan       nan       nan        nan        nan       ] nan
with NuSVR,                         k-value: 5, accuracy: [0.1506646 0.1155647 0.18083134 0.11672907 0.15177754] 0.1431
with OrthogonalMatchingPursuit,     k-value: 5, accuracy: [0.3813879 0.2872013 0.33148025 0.42638247 0.2794747 ] 0.3412
with OrthogonalMatchingPursuitCV,   k-value: 5, accuracy: [0.4393979 0.3732159 0.57089446 0.54241793 0.49723072] 0.4846
with PLSCanonical,                  k-value: 5, accuracy: [-1.28656 -1.822844 -1.9914454 -0.6563578 -0.75678083] -1.3028
with PLSRegression,                 k-value: 5, accuracy: [0.4195822 0.3953964 0.59725774 0.6011566  0.52939695] 0.5086
with PassiveAggressiveRegressor,    k-value: 5, accuracy: [0.4066306 0.3938158 0.51860627 0.52882571 0.51783617] 0.4731
with PoissonRegressor,              k-value: 5, accuracy: [0.3289664 0.2958900 0.37049586 0.40918561 0.36752662] 0.3544
with RANSACRegressor,               k-value: 5, accuracy: [0.1384962 0.208796 -0.41854601 0.3882970 -0.21985824] 0.0194
with RadiusNeighborsRegressor,      k-value: 5, accuracy: [-3.1e-05 -8.59e-03 -8.160e-03 -2.230e-02 -9.9281e-06] -0.0078
with RandomForestRegressor,         k-value: 5, accuracy: [0.4052464 0.3814514 0.5269866  0.51307055 0.41737595] 0.4488
with Ridge,                         k-value: 5, accuracy: [0.3816734 0.3605594 0.4646573  0.4754686  0.44694994] 0.4259
with RidgeCV,                       k-value: 5, accuracy: [0.4357522 0.4203322 0.58365497 0.58118184 0.51909944] 0.508
with SGDRegressor,                  k-value: 5, accuracy: [0.3711795 0.3464226 0.43445539 0.47112037 0.44170082] 0.413
with SVR,                           k-value: 5, accuracy: [0.1591933 0.1316534 0.19968185 0.11371444 0.16149508] 0.1531
with TheilSenRegressor,             k-value: 5, accuracy: [0.4143109 0.4177653 0.56960985 0.56440483 0.51233528] 0.4957
with TransformedTargetRegressor,    k-value: 5, accuracy: [0.4251577 0.4219485 0.59166041 0.58511829 0.51332843] 0.5074
with TweedieRegressor,              k-value: 5, accuracy: [0.006494 -0.001985 -0.0004684 -0.01506812 0.00709943] -0.0008


MultiOutputRegressor will not work for this particular data-set
RegressorChain will not work for this particular data-set
StackingRegressor will not work for this particular data-set
VotingRegressor will not work for this particular data-set


the number of working models: 50
the number of candidate models: 54
'''