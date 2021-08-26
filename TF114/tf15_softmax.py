# tensorflow.nn
#  > Primitive Neural Net (NN) Operations

import tensorflow as tf
tf.set_random_seed(96)

x_data = [[1., 2., 1., 1.], [1., 2., 3., 2.],
	      [3., 1., 3., 4.], [4., 1., 5., 5.],
	      [1., 7., 5., 5.], [1., 2., 5., 6.],
	      [1., 6., 6., 6.], [1., 7., 6., 7.]]

y_data = [[0., 0., 1.], [0., 0., 1.],
          [0., 0., 1.], [0., 1., 0.],
          [0., 1., 0.], [0., 1., 0.],
          [1., 0., 0.], [1., 0., 0.]]

x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 4))
y = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 3))

W = tf.compat.v1.Variable(tf.random_normal([4, 3]))
b = tf.compat.v1.Variable(tf.random_normal([1, 3]))

hypothesis = tf.nn.softmax(tf.matmul(x, W) + b)
objective = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001).minimize(objective)

sss = tf.compat.v1.Session()
sss.run(tf.global_variables_initializer())

for epo in range(8001):
    _, hTS_val, obj_val = sss.run([train, hypothesis, objective], feed_dict={x:x_data, y:y_data})
    if epo % 200 == 0:
        print('for', epo, 'turns,', hTS_val, obj_val)

pred = sss.run(hypothesis, feed_dict={x:[[1., 11., 7., 9.]]})
print(pred,'as one-hot encoded:', sss.run(tf.argmax(pred, axis=1)))
# determine all or none along 1-axis


'''
for 8000 turns, [[3.7959196e-02 1.8929175e-01 7.7274901e-01]
                 [7.4513584e-02 3.6490780e-01 5.6057858e-01]
                 [1.7190247e-03 5.5111974e-01 4.4716129e-01]
                 [2.1107429e-04 5.3710192e-01 4.6268705e-01]
                 [5.6264752e-01 2.9957384e-01 1.3777862e-01]
                 [2.6089144e-01 7.2576535e-01 1.3343130e-02]
                 [5.6551933e-01 3.9320689e-01 4.1273858e-02]
                 [7.0349032e-01 2.8230599e-01 1.4203720e-02]] 
                objective: 0.5888281
prediction: [[0.93315345 0.06497329 0.00187331]] as one-hot encoded: [0]
'''