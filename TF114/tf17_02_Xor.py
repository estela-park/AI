# Building multi-layered Perceptron
import tensorflow as tf


x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_normal([2, 12]))
b1 = tf.Variable(tf.random_normal([12]))

W2 = tf.Variable(tf.random_normal([12, 10]))
b2 = tf.Variable(tf.random_normal([10]))

W4 = tf.Variable(tf.random_normal([10, 1], 0, 0.01))
b4 = tf.Variable(tf.random_normal([1], 0, 0.01))

hidden1 = tf.sigmoid(tf.matmul(x, W1) + b1)
hidden2 = tf.sigmoid(tf.matmul(hidden1, W2) + b2)
hypothesis = tf.sigmoid(tf.matmul(hidden2, W4) + b4)

cost = -tf.reduce_mean(y*tf.log(hypothesis) + (1 - y)*tf.log(1 - hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=2e-2)
train = optimizer.minimize(cost)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))

sss = tf.compat.v1.Session()
sss.run(tf.global_variables_initializer())

for epoch in range(8001):
    cost_val, hTS_val, _ = sss.run([cost, hypothesis, train],
                                feed_dict={x: x_data, y: y_data})
    if epoch % 100 ==0:
        print(epoch, cost_val, hTS_val)

h, c, a = sss.run([hypothesis, predict, accuracy], feed_dict={x: x_data, y: y_data})
print(h, c, a)

'''
2 hidden-layer-deep, learning-rate = 2e-2, epochs=8000
[[0.] [1.] [1.] [0.]], accuracy: 1.0
'''