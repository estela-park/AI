from numpy.core.fromnumeric import shape
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), _ = mnist.load_data()

# GAN uses tanh for its Generator's final activation fn, 
# and tanh's product ranges from -1 to 1.
# x_train = [0, 255]
x_train = x_train / 127.5 - 1

# Generator take 1D noise vector as its input
x_train = x_train.reshape(60000, 28*28)

# nose vector's lenth can vary
NOISE_DIM = 10

# for vanilla GAN and DCGAN, 
# setting learning_rate = 0.0002, beta_1=0.5 is known to be effective.
adam = Adam(lr=0.0002, beta_1=0.5)

# Generator definition
generator = Sequential([
    Dense(256, input_dim=NOISE_DIM),
    LeakyReLU(0.2),
    Dense(512),
    LeakyReLU(0.2),
    Dense(1024),
    LeakyReLU(0.2),
    Dense(28*28, activation='tanh')
])

# Discriminator definition
discriminator = Sequential([
    # discriminator's input is generator's output
    # kernel_initializer initializes Ws of Disc. according to its arg.
    Dense(1024, input_shape=(28*28,), kernel_initializer=RandomNormal(stddev=0.02)),
    LeakyReLU(0.2),
    Dropout(0.3),
    Dense(512),
    LeakyReLU(0.2),
    Dropout(0.3),
    Dense(256),
    LeakyReLU(0.2),
    Dropout(0.3),
    # Discriminator's output is probability of its input's being authentic image
    Dense(1, activation='sigmoid')
])

# Discriminator should be compiled
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

# Connecting Generator with Discriminator
discriminator.trainable = False
gan_input = Input(shape=(NOISE_DIM, ))
# x will be the image(serialized) the Generator generates
x = generator(inputs=gan_input)
# output will be the Discriminator's evaluation of x's image
output = discriminator(x)

# Model combines Generator and Discriminator
gan = Model(gan_input, output)
gan.compile(loss='binary_crossentropy', optimizer=adam)


# fn that takes data and packages it into batches
def get_batches(data, batch_size):
    batches = []

    # divide the number of data-points with batch_size
    # equals epochs
    for i in range(int(data.shape[0] // batch_size)):
        batch = data[i*batch_size: (i+1)*batch_size]
        batches.append(batch)
    return np.asarray(batches)


def visualize_training(epoch, d_losses, g_losses):
    # visualize the losses of G and D
    plt.figure(figsize=(8, 4))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    print('epoch: {}, D-loss: {}, G-loss: {}'.format(epoch, np.asarray(d_losses).mean(), np.asarray(g_losses).mean()))

    # visualize the creation of the Generator
    # 24 images
    noise = np.random.normal(0, 1, size=(24, NOISE_DIM))
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(-1, 28, 28)

    plt.figure(figsize=(8, 4))
    for i in range(generated_images.shape[0]):
        plt.subplot(4, 6, i+1)
        plt.imshow(generated_images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Training
BATCH_SIZE = 128
EPOCHS = 50

d_losses = []
g_losses = []

for epoch in range(1, EPOCHS + 1):

    for real_images in get_batches(x_train, BATCH_SIZE):
        input_noise = np.random.uniform(-1, 1, size=[BATCH_SIZE, NOISE_DIM])
        # np.random.uniform(low, high, size)
        generated_images = generator.predict(input_noise)
        x_dis = np.concatenate([real_images, generated_images])
        # print('raw.shape: {}, G_generated.shape: {}, x_dis.shape: {}'.format(real_images.shape, generated_images.shape, x_dis.shape))

        # assigning accuracy to the generated images 0, real images 0.9
        y_dis = np.zeros(2*BATCH_SIZE)
        y_dis[:BATCH_SIZE] = 0.9

        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(x_dis, y_dis)

        noise = np.random.uniform(-1, 1, size=[BATCH_SIZE, NOISE_DIM])
        y_gan = np.ones(BATCH_SIZE)

        # training is not done on the Generator alone, 
        # G needs noise vectors for inputs serialized images for outputs
        # whole gan model should be trained but D's Ws to be frozen

        # .train_on_batch(): runs a single gradient update on a single batch of data.
        #                    returns scalar training loss or list of scalars
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, y_gan)
    d_losses.append(d_loss)
    g_losses.append(g_loss)

    if epoch == 1 or epoch % 5 == 0:
        visualize_training(epoch, d_losses, g_losses)