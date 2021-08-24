import tensorflow as tf
sss = tf.Session()

x = tf.Variable([2], dtype=tf.float32, name='test')
# name keyarg is optional, user calls the variable x

init = tf.global_variables_initializer()

# every execution in tf.ver1 should be run by session
sss.run(init)
print(sss.run(x))