import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sss = tf.Session()
print(node3)
# Tensor("Add:0", shape=(), dtype=float32)
print(sss.run(node3))
# 7.0