from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes, load_wine
from tensorflow.keras.datasets import mnist, cifar10, cifar100
import numpy as np

datasets = load_iris()
x_data_iris = datasets.data   
y_data_iris = datasets.target 

np.save('./_save/_npy/k55_x_data_iris.npy', arr=x_data_iris)
np.save('./_save/_npy/k55_y_data_iris.npy', arr=y_data_iris)

datasets = load_boston()
x_data_boston = datasets.data 
y_data_boston = datasets.target

np.save('./_save/_npy/k55_x_data_boston.npy', arr=x_data_boston)
np.save('./_save/_npy/k55_y_data_boston.npy', arr=y_data_boston)

datasets = load_breast_cancer()
x_data_cancer = datasets.data  
y_data_cancer = datasets.target

np.save('./_save/_npy/k55_x_data_cancer.npy', arr=x_data_cancer)
np.save('./_save/_npy/k55_y_data_cancer.npy', arr=y_data_cancer)

datasets = load_diabetes()
x_data_diabetes = datasets.data   
y_data_diabetes = datasets.target 

np.save('./_save/_npy/k55_x_data_diabetes.npy', arr=x_data_diabetes)
np.save('./_save/_npy/k55_y_data_diabetes.npy', arr=y_data_diabetes)

datasets = load_wine()
x_data_wine = datasets.data   
y_data_wine = datasets.target 

np.save('./_save/_npy/k55_x_data_wine.npy', arr=x_data_wine)
np.save('./_save/_npy/k55_y_data_wine.npy', arr=y_data_wine)


(x_train, y_train), (x_test, y_test) = mnist.load_data() 
# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x_data_mnist = np.array(x_train, x_test)
y_data_mnist = np.array(y_train, y_test)

np.save('./_save/_npy/k55_x_data_mnist.npy', arr=x_data_mnist)
np.save('./_save/_npy/k55_y_data_mnist.npy', arr=y_data_mnist)

(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
# (50000, 32, 32, 3) (10000, 32, 32, 3)
# (50000, 1) (10000, 1)

x_data_cifar10 = np.array(x_train, x_test)
y_data_cifar10 = np.array(y_train, y_test)

np.save('./_save/_npy/k55_x_data_mnist.npy', arr=x_data_cifar10)
np.save('./_save/_npy/k55_y_data_mnist.npy', arr=y_data_cifar10)

(x_train, y_train), (x_test, y_test) = cifar100.load_data() 
# (50000, 32, 32, 3) (10000, 32, 32, 3)
# (50000, 1) (10000, 1)

x_data_cifar100 = np.array(x_train, x_test)
y_data_cifar100 = np.array(y_train, y_test)

np.save('./_save/_npy/k55_x_data_mnist.npy', arr=x_data_cifar100)
np.save('./_save/_npy/k55_y_data_mnist.npy', arr=y_data_cifar100)