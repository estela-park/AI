import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

tf.compat.v1.set_random_seed(96)

dataset = load_diabetes()
x_data = dataset.data
# (442, 10)
y_data = dataset.target.reshape(442, 1)
# (442,): [151.  75. 141. 206. 135.  97. 138.  63. 110. 310.]


x_data, x_test, y_data, y_test = train_test_split(x_data, y_data, train_size=0.9, random_state=84)
# 397/45

x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 10))
y = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 1))

W = tf.compat.v1.Variable(tf.random_normal([10, 1], dtype=tf.float32))
# Wa = tf.compat.v1.Variable(tf.random_normal([1], dtype=tf.float32))
b = tf.compat.v1.Variable(tf.random_normal([1], dtype=tf.float32))
# ba = tf.compat.v1.Variable(tf.random_normal([1], dtype=tf.float32))

# hypothesis = tf.maximum(0., tf.matmul(x, W) + b)
# not trainable

# hypothesis = tf.maximum(0., tf.matmul(x, W) + b)
# hypothesisa = hypothesis * Wa + ba
# 83190 turns cost: 5744.6167 -> froze there

hypothesis = tf.matmul(x, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimizer.minimize(cost)

sss = tf.compat.v1.Session()
sss.run(tf.global_variables_initializer())

for epo in range(83200):
    cost_val, pred_val, _ = sss.run([cost, hypothesis, train], feed_dict={x:x_data, y:y_data})
    if epo % 30 == 0:
        print(epo,'turns cost:',cost_val)

print('R2:', r2_score(y_test, sss.run(hypothesis, feed_dict={x:x_test})))
sss.close()


'''
LearngRate=0.0005
83190 turns cost: 3939.028
R2: 0.25088578268385375

LearngRate=0.001
83190 turns cost: 3366.69
R2: 0.32645798504198076

LearngRate=0.005
83190 turns cost: 2790.0322
R2: 0.44095208054698243

LearngRate=0.01
83190 turns cost: 2760.9912
R2: 0.45875934762404913

LearngRate=0.05
83190 turns cost: 2753.104
R2: 0.46141970437943747
'''