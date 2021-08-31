import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# 1. Data-prop
tf.set_random_seed(72)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)

x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

# 2. Modelling
W1 = tf.compat.v1.Variable(tf.random.normal([3, 3, 1, 32], 0, 3e-3), name='filter1_as_kernel_in_channel_out_channel')


learning_rate = 0.00005
training_epochs = 10
batch_size = 100
total_batch = int(len(x_train)//batch_size)

# 2-1 Layer1
L1 = tf.compat.v1.nn.conv2d(x, W1, [1, 1, 1, 1], 'SAME')
L1 = tf.compat.v1.nn.relu(L1)
L1_Mpooled = tf.compat.v1.nn.max_pool2d(L1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
# Tensor("MaxPool2d:0", shape=(?, 14, 14, 32), dtype=float32)

# 2-2 Layer2
W2 = tf.compat.v1.Variable(tf.random.normal([3, 3, 32, 64], 0, 3e-3), name='filter2')
L2 = tf.compat.v1.nn.conv2d(L1_Mpooled, W2, [1, 1, 1, 1], 'SAME')
L2 = tf.compat.v1.nn.selu(L2)
L2_Mpooled = tf.compat.v1.nn.max_pool2d(L2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
# Tensor("MaxPool2d_1:0", shape=(?, 7, 7, 64), dtype=float32)

# 2-3 Layer3
W3 = tf.compat.v1.Variable(tf.random.normal([3, 3, 64, 128], 0, 3e-3), name='filter3')
L3 = tf.compat.v1.nn.conv2d(L2_Mpooled, W3, [1, 1, 1, 1], 'SAME')
L3 = tf.compat.v1.nn.elu(L3)
L3_Mpooled = tf.compat.v1.nn.max_pool2d(L3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
# Tensor("MaxPool2d_2:0", shape=(?, 4, 4, 128), dtype=float32)

# 2-4 Layer4
W4 = tf.compat.v1.Variable(tf.random.normal([3, 3, 128, 64], 0, 3e-3),  name='filter4')
L4 = tf.compat.v1.nn.conv2d(L3_Mpooled, W4, [1, 1, 1, 1], 'SAME')
L4 = tf.compat.v1.nn.leaky_relu(L4)
L4_Mpooled = tf.compat.v1.nn.max_pool2d(L4, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
# Tensor("MaxPool2d_3:0", shape=(?, 2, 2, 64), dtype=float32)

# 2-5 Layer5: flattening
L_f = tf.compat.v1.reshape(L4_Mpooled, [-1, 2*2*64])
# Tensor("Reshape:0", shape=(?, 256), dtype=float32)

# 2-6 Layer6: DNN
W6 = tf.compat.v1.Variable(tf.random.normal([2*2*64, 64], 0, 3e-3), name='Weight6')
b6 = tf.compat.v1.Variable(tf.random.normal([64], 0, 3e-3), name='bias6')
L6 = tf.compat.v1.matmul(L_f, W6) + b6
L6 = tf.compat.v1.nn.selu(L6)
L6 = tf.compat.v1.nn.dropout(L6, keep_prob=0.8)
# Tensor("dropout/mul_1:0", shape=(?, 64), dtype=float32)

# 2-7 Layer7: DNN
W7 = tf.compat.v1.Variable(tf.random.normal([64, 32], 0, 3e-3), name='Weight7')
b7 = tf.compat.v1.Variable(tf.random.normal([32], 0, 3e-3), name='bias7')
L7 = tf.compat.v1.matmul(L6, W7) + b7
L7 = tf.compat.v1.nn.selu(L7)
L7 = tf.compat.v1.nn.dropout(L7, 0.8)
# ensor("dropout_1/mul_1:0", shape=(?, 32), dtype=float32)

# 2-8 Layer8: softmax
W8 = tf.compat.v1.Variable(tf.random.normal([32, 10], 0, 3e-3), name='softmax_Weight')
b8 = tf.compat.v1.Variable(tf.random.normal([10], 0, 3e-3), name='softmax_bias')
L8 = tf.compat.v1.matmul(L7, W8) + b8
# Tensor("add_2:0", shape=(?, 10), dtype=float32)
hypothesis = tf.compat.v1.nn.softmax(L8)
# Tensor("Softmax:0", shape=(?, 10), dtype=float32)

loss = tf.compat.v1.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# 3. Training
sss = tf.compat.v1.Session()
sss.run(tf.compat.v1.global_variables_initializer())

for epoch in range(training_epochs):
    avg_loss = 0
    for i in range(total_batch):
        start = i*batch_size
        end = start + batch_size
        batch_x, batch_y = x_train[start:end], y_train[start:end]

        feed_dict = {x:batch_x, y:batch_y}
        batch_loss, _ = sss.run([loss, optimizer], feed_dict=feed_dict)
        avg_loss += batch_loss/total_batch

    print('Epoch:', '%04d' %(epoch + 1), 'loss: {:.9f}'.format(avg_loss))

pred = sss.run(hypothesis, feed_dict={x:x_test})
acc = accuracy_score(y_test, np.argmax(pred, axis=1))
print('accuracy:', acc)
# accuracy: 0.7325