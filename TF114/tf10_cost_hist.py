import tensorflow as tf
import matplotlib.pyplot as plt

x = [1., 2., 3.]
y = [2., 4., 6.]

W = tf.placeholder(tf.float32)

hypothesis = x*W
# multiplying element-wise abide by LA rules


cost = tf.reduce_mean(tf.square(hypothesis - y))
# Linear Algebra: reduced echelon matrics
# Computes the mean of elements across dimensions of a tensor.
# Reduces input_tensor along the dimensions given in axis 
# by computing the mean of elements across the dimensions in axis. 
# Unless keepdims is true(default: keepdims=False), 
# the rank of the tensor is reduced by 1 for each of the entries in axis, which must be unique. 


W_hist = []
cost_hist = []

with tf.compat.v1.Session() as sss:
    for i in range(-30, 50):
        sss.run(tf.compat.v1.global_variables_initializer())
        currentW = i
        currentCost = sss.run(cost, feed_dict={W: currentW})
        
        W_hist.append(currentW)
        cost_hist.append(currentCost)

print('====================Weight History====================')
print(W_hist)
print('====================Cost History====================')
print(cost_hist)

plt.plot(W_hist, cost_hist)
plt.xlabel('Weight')
plt.ylabel('Object')
plt.show()