# the winter of AI
import tensorflow as tf


x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, W) + b)
cost = -tf.reduce_mean(y*tf.log(hypothesis) + (1 - y)*tf.log(1 - hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))

sss = tf.compat.v1.Session()
sss.run(tf.global_variables_initializer())

for epoch in range(2001):
    cost_val, hTS_val, _ = sss.run([cost, hypothesis, train],
                                feed_dict={x: x_data, y: y_data})
    if epoch % 10 ==0:
        print(epoch, cost_val, hTS_val)

h, c, a = sss.run([hypothesis, predict, accuracy], feed_dict={x: x_data, y: y_data})
print(h, c, a)