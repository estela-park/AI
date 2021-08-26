import tensorflow as tf
tf.set_random_seed(78)

x1_data = [72., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([1]), name='weight1')
W2 = tf.Variable(tf.random_normal([1]), name='weight2')
W3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1*W1 + x2*W2 + x3*W3 + b

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000045)
# learning rate should be much smaller
train = optimizer.minimize(cost)

sss = tf.compat.v1.Session()
sss.run(tf.global_variables_initializer())

for epoch in range(2001):
    cost_val, hTS_val, _ = sss.run([cost, hypothesis, train],
                                feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data})
    if epoch % 10 ==0:
        print(epoch, cost_val, hTS_val)


'''
2000 0.65880805 [152.2204  183.83545 181.34895 195.7861  141.84554]
learning rate max: 0.000045
@0.000048 it crushes
'''