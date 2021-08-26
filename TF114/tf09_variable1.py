# many ways to open a session

import tensorflow as tf
tf.compat.v1.set_random_seed(164)

W = tf.Variable(tf.random_normal([1]), name='weight')


# plain session
sss = tf.compat.v1.Session()
sss.run(tf.global_variables_initializer())
print(sss.run(W))
sss.close()


# interactive session, variable should be eval() should be called,
# this should compute variable and return the value accordingly.
sss = tf.InteractiveSession()
sss.run(tf.global_variables_initializer())
print(W.eval())
sss.close()


# plain session with eval()
sss = tf.Session()
sss.run(tf.global_variables_initializer())
print(W.eval(sss))
print(W.eval(session=sss))
sss.close()

'''
[-0.2551619]
[-0.2551619]
[-0.2551619]
[-0.2551619]
'''