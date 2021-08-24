import tensorflow as tf

node1 = tf.constant(13.0)
node2 = tf.constant(7.0)
nodeAdd = tf.add(node1, node2)

sss = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

additionNode = a + b
print(sss.run(additionNode, feed_dict={a: 3, b: 4.5}))
print(sss.run(additionNode, feed_dict={a: [3, 1], b: [4.5, 8]}))

additionNtriple = additionNode * 3
print(sss.run(additionNtriple, feed_dict={a: 3, b: 4.5}))
