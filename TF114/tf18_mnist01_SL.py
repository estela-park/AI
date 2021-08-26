import tensorflow as tf
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# (60000, 28, 28) (60000,)
# np.unique(y_train): [0 1 2 3 4 5 6 7 8 9]

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
y_train = y_train.reshape(60000, 1)
y_test = y_test.reshape(10000, 1)

enc = OneHotEncoder()
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()

x = tf.placeholder(tf.float32, shape=[None, 28*28])
y = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.random.normal([28*28, 10], 0, 1e-3), name='weight')
b = tf.Variable(tf.random.normal([10], 0, 1e-3), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x, W) + b)
objective = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(objective)

sss = tf.compat.v1.Session()
sss.run(tf.compat.v1.global_variables_initializer())

for epo in range(101):
    h, ob = sss.run([hypothesis, objective], feed_dict={x:x_train, y:y_train})
    _, hTS_val, obj_val = sss.run([train, hypothesis, objective], feed_dict={x:x_train, y:y_train})
    if epo % 10 == 0:
        print('for', epo, 'turns,', hTS_val, obj_val)


# can't handle a mix of multilabel-indicator and continuous-multioutput targets
# Do the one-hot encoding!
pred = tf.argmax(sss.run(hypothesis, feed_dict={x:x_test}), axis=1).eval(session=sss)
print('as classified:', pred)
print('accuracy:', accuracy_score(y_test, enc.transform(pred.reshape(-1, 1))))

'''
as classified: [6 1 6 ... 6 6 6]
accuracy: 0.0785

learning_rate=1e-5
as classified: [7 2 1 ... 4 8 6]
accuracy: 0.8805
'''