import tensorflow as tf
tf.set_random_seed(78)
# work as random_state

x_train = [1, 2, 3]
y_train = [3, 5, 7]

W = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

hypothesis = W*x_train + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sss = tf.Session()
sss.run(tf.global_variables_initializer())

for step in range(2001):
    sss.run(train)
    if step % 20 == 0:
        print(step, sss.run(loss), sss.run(W), sss.run(b))

