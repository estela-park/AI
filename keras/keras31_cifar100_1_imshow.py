import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100


(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

print(x_train[37])
print("y[0] value", y_train[37]) 

plt.imshow(x_train[72], 'gray')
plt.show()