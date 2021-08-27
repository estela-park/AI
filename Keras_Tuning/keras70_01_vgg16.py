# The ImageNet is a dataset which contains 14,197,122 annotated images.
# It is used for image classification(with 1000 classes) and object detection.
#
# VGG16 is a convolutional neural network model for Large-Scale Image Recognition. 
# The model achieves 92.7% top-5 test accuracy in ImageNet.


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet', include_top=True)
# include_top=
#            -True: Default, Input layer immutable and  include the 3 FC layers at the top of the network.
#            -False: input shape and FC can be altered
#          > input_shape: optional shape tuple, 
#            only to be specified if include_top is False 
#            (otherwise the input shape has to be (224, 224, 3)
# weights: None (random initialization) | 'imagenet' | the path to the weights file


# model.trainable=False: Weights trained with imagenet are used without update
# len(model.weights)=26
# len(model.trainabel_weights)=26|0

# include_top=True
# len(model.weights)=32

# input_1 (InputLayer)         [(None, 224, 224, 3)]     0
# ** to train 32, 32, 3 images, the images have to be altered, 
#    or VGG16's input should be manipulated.
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
# .................................................................
# flatten (Flatten)            (None, 25088)             0
# _________________________________________________________________
# fc1 (Dense)                  (None, 4096)              102764544
# _________________________________________________________________
# fc2 (Dense)                  (None, 4096)              16781312
# _________________________________________________________________
# predictions (Dense)          (None, 1000)              4097000
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0


# Fully Connected: Layers where all the inputs from a layer are connected to every next layer. 
#                  In most popular machine learning models, 
#                  the last few layers are full connected layers which compiles 
#                  the data extracted by previous layers to form the final output. 
#                  It is the second most time consuming layer second to Convolution Layer.
#                   > CNN can't be FCed, FC exists as a part of DNN structure.
# VS Dropout: Nodes from a layer are randomly excluded from the calculations.