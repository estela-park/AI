import tensorflow as tf
tf.set_random_seed(78)

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
# shape=None: tf automatically shapes it
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

W = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)

hypothesis = W*x_train + b

loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis - y_train))
# calculatig loss requires value from y_train; placeholder
# it should be run by session to be operated

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sss = tf.compat.v1.Session()
sss.run(tf.compat.v1.global_variables_initializer())

for step in range(2001):
    _, loss_val = sss.run([train, loss], feed_dict={x_train:[1, 2, 3], y_train:[1, 2, 3]})
    # train doesn't need to be specified by return 
    if step % 20 == 0:
        print(step, loss_val, sss.run(W), sss.run(b))

    # W and b can be run parallel to train and loss
    # _, loss_val, W_val, b_val = sss.run([train, loss, W, b], feed_dict={x_train:[1, 2, 3], y_train:[1, 2, 3]})
    # if step % 20 == 0:
    #     print(step, loss_val, W_val, b_val)

'''
1960 4.1102685e-05 [1.0074282] [-0.01688612]
1980 3.732971e-05 [1.0070792] [-0.01609252]
2000 3.3904245e-05 [1.0067466] [-0.01533634]
'''