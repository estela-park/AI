import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

tf.compat.v1.set_random_seed(96)

dataset = load_boston()
x_data = dataset.data
y_data = dataset.target.reshape(506, 1)
# (506, 13), (506, 1)

x_data, x_test, y_data, y_test = train_test_split(x_data, y_data, train_size=0.9, random_state=84)
print(x_data.shape)
print(y_data.shape)
print(x_test.shape)
print(y_test.shape)
x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 13))
y = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 1))

W = tf.compat.v1.Variable(tf.random_normal([13, 1], dtype=tf.float32))
b = tf.compat.v1.Variable(tf.random_normal([1], dtype=tf.float32))

hypothesis = tf.matmul(x, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000003)
train = optimizer.minimize(cost)

sss = tf.compat.v1.Session()
sss.run(tf.global_variables_initializer())

for epo in range(83200):
    cost_val, pred_val, _ = sss.run([cost, hypothesis, train], feed_dict={x:x_data, y:y_data})
    if epo % 30 == 0:
        print(epo,'turns cost:',cost_val)


print('R^2 score:',r2_score(y_test, sss.run(hypothesis, feed_dict={x:x_test})))

'''
(learning_rate=0.000003)
83190 turns cost: 34.91149
R^2 score: 0.3380133004056315
'''