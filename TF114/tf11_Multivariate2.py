import tensorflow as tf
tf.compat.v1.set_random_seed(78)

x_data = [[73, 51, 65],
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]
y_data = [[152], [185], [180], [205], [142]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random.normal([3, 1]))
b = tf.Variable(tf.random.normal([1]))

hypothesis = tf.matmul(x, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00008)
train = optimizer.minimize(cost)

sss = tf.compat.v1.Session()
sss.run(tf.global_variables_initializer())

for epoch in range(2001):
    cost_val, hTS_val, _ = sss.run([cost, hypothesis, train],
                                feed_dict={x: x_data, y: y_data})
    if epoch % 10 ==0:
        print(epoch, cost_val, hTS_val)


'''
2000 307.3898 [[171.2223 ] [191.5163 ] [151.22272] [213.23962] [126.86836]]
LR @0.00008
'''