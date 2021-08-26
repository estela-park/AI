import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

print(x_train[3])
print("y[0] value", y_train[3]) 

plt.imshow(x_train[3], 'gray')
plt.show()