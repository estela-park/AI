# tf.nn.relu/ elu/ selu/ dropout are supported

import tensorflow as tf
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
tf.set_random_seed(72)

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

# images are in black and white, either 0 or 1 
W1 = tf.Variable(tf.random_normal([28*28, 28*14], 0, 3e-2))
b1 = tf.Variable(tf.random_normal([28*14], 0, 1e-2))

W2 = tf.Variable(tf.random_normal([28*14, 28*7], 0, 3e-2))
b2 = tf.Variable(tf.random_normal([28*7], 0, 1e-2))

W3 = tf.Variable(tf.random_normal([28*7, 28], 0, 3e-2))
b3 = tf.Variable(tf.random_normal([28], 0, 1e-2))

W4 = tf.Variable(tf.random_normal([28, 10], 0, 3e-2))
b4 = tf.Variable(tf.random_normal([10], 0, 1e-2))

hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden2 = tf.nn.selu(tf.matmul(hidden1, W2) + b2)
hidden3 = tf.nn.selu(tf.matmul(hidden2, W3) + b3)
hypothesis = tf.nn.softmax(tf.matmul(hidden3, W4) + b4)

objective = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))

optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=35e-2)
train = optimizer.minimize(objective)

sss = tf.compat.v1.Session()
sss.run(tf.compat.v1.global_variables_initializer())

for epo in range(351):
    h, ob = sss.run([hypothesis, objective], feed_dict={x:x_train, y:y_train})
    _, hTS_val, obj_val = sss.run([train, hypothesis, objective], feed_dict={x:x_train, y:y_train})
    if epo % 10 == 0:
        print('for', epo, 'turns,', hTS_val, obj_val)

pred = tf.argmax(sss.run(hypothesis, feed_dict={x:x_test}), axis=1).eval(session=sss)
print('as classified:', pred)
print('accuracy:', accuracy_score(y_test, enc.transform(pred.reshape(-1, 1))))

'''
as classified: [7 2 1 ... 4 5 6]
accuracy: 0.901

@random=72
as classified: [7 2 1 ... 4 5 6]
accuracy: 0.9196

all the initial-biases to be 0, 1e-1
as classified: [7 2 1 ... 4 5 6]
accuracy: 0.9209

all the initial-weights to be 0, 3e-2
as classified: [7 2 1 ... 4 5 6]
accuracy: 0.9283

learning-rate=27e-2
as classified: [7 2 1 ... 4 5 6]
accuracy: 0.9335

epochs=1001
as classified: [7 2 1 ... 4 5 6]
accuracy: 0.9775
'''