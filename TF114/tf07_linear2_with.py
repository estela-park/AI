import tensorflow as tf
tf.set_random_seed(66)

x_train = [1, 2, 3]
y_train = [3, 5, 7]

W = tf.Variable([1], dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32)

hypothesis = W*x_train + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

with tf.Session() as sss:
    sss.run(tf.global_variables_initializer())

    for step in range(2001):
        sss.run(train)
        if step % 20 == 0:
            print(step, sss.run(loss), sss.run(W), sss.run(b))

'''
1960 1.54997e-06 [1.9985539] 1.0032872
1980 1.408174e-06 [1.9986217] 1.0031328
2000 1.2789059e-06 [1.9986867] 1.0029857
'''