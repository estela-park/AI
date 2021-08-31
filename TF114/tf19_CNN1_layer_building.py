import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical


tf.set_random_seed(72)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255


W1 = tf.compat.v1.get_variable('filter1_as_kernel_in_channel_out_channel', shape=[3, 3, 1, 32])

sss = tf.compat.v1.Session()
sss.run(tf.compat.v1.global_variables_initializer())
# initializing is not giving initial values, 
# but assigning tf.value to its memory space.

# np.min(sss.run(W1)): -0.14158298
# np.max(sss.run(W1)): 0.14206983
# np.mean(sss.run(W1)): 0.007303242
# np.median(sss.run(W1)): 0.0021756291

# Layer
L1 = tf.compat.v1.nn.conv2d(x, W1, [1, 1, 1, 1], 'SAME')
# when giving args positionally, input, filters
# Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
L1 = tf.compat.v1.nn.relu(L1)
# Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
L1_Mpooled = tf.compat.v1.nn.max_pool2d(L1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
# Tensor("MaxPool2d:0", shape=(?, 14, 14, 32), dtype=float32)