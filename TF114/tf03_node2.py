import tensorflow as tf

node1 = tf.constant(13.0)
node2 = tf.constant(7.0)
nodeAdd = tf.add(node1, node2)
nodeSubt = tf.subtract(node1, node2)
nodeMult = tf.multiply(node1, node2)
nodeDiv = tf.divide(node1, node2)

sss = tf.Session()

print(sss.run(nodeAdd))
print(sss.run(nodeSubt))
print(sss.run(nodeMult))
print(sss.run(nodeDiv))