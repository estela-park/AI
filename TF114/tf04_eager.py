# TensorFlow's eager execution is an imperative programming environment 
# that evaluates operations immediately, without building graphs: 
# operations return concrete values instead of constructing a computational graph to run later. 

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
print(tf.__version__)

hello = tf.constant('Hello world')
print(hello)

sess = tf.compat.v1.Session()
print(sess.run(hello))