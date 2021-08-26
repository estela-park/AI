from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

# x:  (50000, 32, 32, 3) (10000, 32, 32, 3)
# y:  (50000, 1) (10000, 1)

print(x_train[3])
print("y[0] value", y_train[3]) 

plt.imshow(x_train[2])
plt.show()