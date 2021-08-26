import tensorflow as tf
tf.compat.v1.set_random_seed(164)


x = [1, 2, 3]
W = tf.Variable([.3], tf.float32)
b = tf.Variable([1.], tf.float32)

hypothesis = x*W +b

sss = tf.compat.v1.Session()
sss.run(tf.global_variables_initializer())
print(sss.run(hypothesis))
sss.close()


sss = tf.InteractiveSession()
sss.run(tf.global_variables_initializer())
print(hypothesis.eval())
sss.close()


# plain session with eval()
sss = tf.Session()
sss.run(tf.global_variables_initializer())
# print(hypothesis.eval(sss)): it's suddenly not working?
# did so with W.eval() but the msg goes as follows
# Cannot evaluate tensor using `eval()`: No default session is registered. 
# Use `with sess.as_default()` or pass an explicit session to `eval(session=sess)`
print(hypothesis.eval(session=sss))
sss.close()


'''
[1.3       1.6       1.9000001]
[1.3       1.6       1.9000001]
[1.3       1.6       1.9000001]
'''