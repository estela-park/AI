'''
[Data Augmentation]
 > flow vs flow_from_directory
    -flow: can be used to augment data and save images statically
           -x, y are coupled in <tuple>
    -flow_from_directory: images sorted categorically in its kind's directory
           -x, y are returned in tuple, 
            x_train, y_train = dataget.flow() will do the job

[.flow()]
  flow method returns an iterator which can generate batches of the augmented data.
    this iterator yields tuples of (x, y)
    where x is a list of ndarrays of image data, y is a ndarray of corresponding labels.
  flow(x, y=None, batch_size=32, shuffle=True, sample_weight=None, seed=None, 
       save_to_dir=None, save_prefix='goes to file name', save_format='png', subset=None)
    If 'sample_weight' is not None, the yielded tuples are of the form (x, y, sample_weight). 
    If y is None, only the numpy array x is returned(that is the data won't be labeled)
    subset specifies whether the data generated is for training or validation. 
'''

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.2,
    shear_range=0.5,
    fill_mode='nearest',
)


augment_size = 5
# .flow returns iterator that akes data & label arrays and generates batches of augmented data.
#       it traverse through the values of which number is specified in batch_size=
#       that is it creates range(batch_size) of images
#       using .flow().next() it can returns one image at a time from a batch
x_data = datagen.flow(np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),
                      # numpy.tile(A, reps)
                      #   Construct an array by repeating A the number of times given by reps.
                      #   If reps has length d, the result will have dimension of max(d, A.ndim).
                      #       a = np.array([0, 1, 2])
                      #       np.tile(a, 2)
                      #       >>>array([0, 1, 2, 0, 1, 2])
                      #       np.tile(a, (2, 2))
                      #       >>>array([[0, 1, 2, 0, 1, 2],
                      #                 [0, 1, 2, 0, 1, 2]])

    batch_size=augment_size,
    shuffle=False,
).next()
# w/h next() x_data[i] is a batch of augmented images, (5, 28, 28, 1)
# w/o next() x_data[0] is a tuple of which each entry holds (5, 28, 28, 1), a batch of augmented images


# ImageDataGenerator.flow(x, y, batchz_size=)
# x_data                             : NumpyArrayIterator       :.next()_Tuple
#   > executing .next()_             : x_data[0]: x    /   x_data[1]: label
#                                      (batch, 28, 28, 1)  (batch, )
# x_data[0 ~ # of images / batch]    : Tuple  : train_data_set
# x_data[0 ~ # of images / batch][0] : ndarray: x data
# x_data[0 ~ # of images / batch][1] : ndarray: label


plt.figure(figsize=(7, 7))
for i in range(5):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[i], cmap='gray')

plt.show()