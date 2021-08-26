import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
tf.set_random_seed(78)

dataset = load_breast_cancer()
x_data = dataset.data
# (569, 30)
print(x_data[:2])
y_data = dataset.target.reshape(569, 1)
# [...  0 1 0 1 1 1 1 1 0 0 1  ...] (569, )

x_data, x_test, y_data, y_test = train_test_split(x_data, y_data, train_size=0.9, random_state=78)
# 512/57

x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1])

W = tf.compat.v1.Variable(tf.compat.v1.zeros([30, 1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), dtype=tf.float32)
# tf.zeros: returns a tensor of dtype(default: float32) with shape specified.
# since the input data is miniscule, W and bias should be tiny as well


hypothesis = tf.sigmoid(tf.matmul(x, W) + b)
cost = -tf.reduce_mean(y*tf.log(hypothesis) + (1 - y)*tf.log(1 - hypothesis))
optimizer = tf.train.GradientDescentOptimizer(0.0000001)
train = optimizer.minimize(cost)

sss = tf.compat.v1.Session()
sss.run(tf.compat.v1.global_variables_initializer())
for epo in range(801):
    _, hTS_val, objective_val = sss.run([train, hypothesis, cost], feed_dict={x:x_data, y:y_data})
    if epo % 10 == 0:
        print('for', epo, 'turns,', objective_val)

# Tensor to np.ndarry: tensor-needs-converting.eval(session=session to use)

print('accuracy:', accuracy_score(y_test, (tf.cast(sss.run(hypothesis, feed_dict={x:x_test}) > 0.5, dtype=tf.float32).eval(session=sss))))
# accuracy for binary-classification
#   > tf.reduce_mean(tf.cast(tf.equal(y_actual, y_pred)))

'''
for 800 turns, 0.6371012
accuracy: 0.5263157894736842
'''