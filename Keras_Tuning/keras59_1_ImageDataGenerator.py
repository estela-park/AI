'''
ImageDataGenerator: augments images in real-time while the model is being trained.
  - it randomly transformates images as the images are passed to the model.
      > multiple transformed copies of a single image
  - it results in a robuster model, and spares overhead memory.
  - it gets images from the flow() | flow_from_dataframe() | flow_from_directory() method
  - [Args]
    > directory:	path to the parent folder which contains the subfolder for the different class images.
    > target_size:	size of the input image
    > color_mode:	RGB(default) | grayscale 
    > batch_size:	size of the batches of data
    > class_mode:	binary | categorical(one-hot encoded label)
'''

# With brain images either normal | abnormal
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # rescaling factor. Defaults to None. 
    # If None or 0, no rescaling is applied, 
    # otherwise it multiplies the data by the value provided 
    # (after applying all other transformations)
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # float: fraction of total width, if < 1, or pixels if >= 1.
    rotation_range=5,
    zoom_range=1.2,
    # Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range]
    shear_range=0.7,
    # refers to the contact point between the lower shear blade and the  upper shear blade
    # Shear Intensity (Shear angle in counter-clockwise direction in degrees)
    fill_mode='nearest',
 # <more args>
 # target_size: Tuple of integers (height, width), defaults to (256,256). 
 #              The dimensions to which all images found will be resized.
 # seed: set the random_state for shuffling and transformations to reproduce the resultrandom_state
)


test_datagen = ImageDataGenerator(rescale=1./255)

# flow_from_directory: every image in directory is loaded.
# there are sub-directories of which names are used as labels.
xy_train = train_datagen.flow_from_directory(
    '../_data/brain/train/',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    # default: shuffle=True,
    # default: color_mode='rgb'
)
# Found 160 images belonging to 2 classes.
# xy_train                                                                              : DirectoryIterator
# xy_train[0 ~ # of images/batch]                                                       : Tuple
# x: xy_train[0 ~ # of images/batch][0]: (5, 150, 150, 3): batch, pixcels, color        : ndarray
# y: xy_train[0 ~ # of images/batch][1]: (5, ): batch, binary=scalar|categorical=vector : ndarray

xy_test = test_datagen.flow_from_directory(
    '../_data/brain/test',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    shuffle=True,
)
# Found 120 images belonging to 2 classes.