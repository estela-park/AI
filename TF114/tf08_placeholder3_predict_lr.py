import tensorflow as tf
tf.set_random_seed(66)

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

W = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)

hypothesis = W*x_train + b

loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis - y_train))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimizer.minimize(loss)

sss = tf.compat.v1.Session()
sss.run(tf.compat.v1.global_variables_initializer())

W_val, b_val = 0, 0
for step in range(100):
    _, loss_val, W_val, b_val = sss.run([train, loss, W, b], feed_dict={x_train:[1, 2, 3], y_train:[3, 5, 7]})
    if step % 10 == 0:
        print(step, loss_val, W_val, b_val)
        # W, b: tensor.RefVariable
        # W_val, b_val: np.ndarray
print('Training done, W:', W_val, '& b:', b_val)

W_val = tf.constant(W_val)
b_val = tf.constant(b_val)

testNode1 = tf.constant([2.])
testNode2 = tf.constant([3., 4.])
testNode3 = tf.constant([5., 6., 7.])

predNode1 = W_val*testNode1 + b_val
predNode2 = W_val*testNode2 + b_val
predNode3 = W_val*testNode3 + b_val

print(sss.run(predNode1))
print(sss.run(predNode2))
print(sss.run(predNode3))

'''
Training done, W: [1.9611211] & b: [1.0883809]
[5.010623]
[6.971744 8.932865]
[10.893986 12.855107 14.816229]
'''