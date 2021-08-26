import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


dataset = load_iris()
x_data = dataset.data
# (150, 4) [0 ~ 9]
y_data = dataset.target.reshape(150, 1)
# (150, ) [0, 1, 2]

enc = OneHotEncoder()
y_data = enc.fit_transform(y_data).toarray()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=78)

x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 4))
y = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 3))

W = tf.compat.v1.Variable(tf.random.normal([4, 3], 0, 1e-2))
b = tf.compat.v1.Variable(tf.random.normal([1, 3], 0, 1e-1))

hypothesis = tf.nn.softmax(tf.matmul(x, W) + b)
objective = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1))

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(objective)

sss = tf.compat.v1.Session()
sss.run(tf.compat.v1.global_variables_initializer())

for epo in range(5001):
    _, hTS_val, obj_val = sss.run([train, hypothesis, objective], feed_dict={x:x_train, y:y_train})
    if epo % 80 == 0:
        print('for', epo, 'turns,', hTS_val, obj_val)

pred = tf.argmax(sss.run(hypothesis, feed_dict={x:x_test}), axis=1).eval(session=sss)
print('as classified:', pred)
print('accuracy:', accuracy_score(y_test, enc.transform(pred.reshape(-1, 1))))


'''
learning-rate=1e-3
as classified: [2 2 2 1 1 1 1 0 1 2 2 2 2 0 1]
accuracy: 0.9333333333333333
'''