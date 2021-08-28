# **For TensorFlow 2.2+, the Keras fit() method now supports generators, 
#   fit_generator might be removed in a future version of Keras.

'''
ImageDataGenerator
  [Methods]
   1 * fit		    | train the model on an array of image data
	    1. entire training data fit into RAM
	       that is there is no need to move old batches of data out of RAM and move new batches of data into RAM
	    2. no data augmentation, model is trained on the given raw data  
	       that is there is no need to manipulate the training data on the fly while training
   2 * fit_generator| train the model on a generator(ImageDataGenerator)
		              fits(trains) the model at the same time generaing the data for the model
	    1. training data too large to fit into memory
	    2. to avoid overfitting, data augmentation is required
	        ++ how it works ++
	         1. fit_generator calls the generator
	         2. the generator passes the batch size to the fit_generator
	         3. with given batch_size The .fit_generator performs backpropagation, and updates the weights
	         4. 1-3 steps are repeated until the desired number of epochs is reached
	        ++ Args ++
	         1. epochs: 		the number of forward/backward passes of the training data
	         2. steps_per_epoch:the number of batches of images that are in a single epoch. 
			                    usually the size of the raw data divided by the batch size.
			                + data generator is meant to loop infinitely
			                + fit_generator needs to determine when one epoch ends and new one begins
			                + once fit_generator hits step_per_epoch, it knows to start a new epoch
			                + while generator supplies infinite amount data to be trained for fit_generator
	        3. validation_steps: steps_per_epoch for validation data. 
				                 only used augmenting the validation set images as well
   3 * train_on_batch
'''

# With brain images either normal | abnormal
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
    # One of {"constant", "nearest", "reflect" or "wrap"}. Default is 'nearest'. 
    # Points outside the boundaries of the input are filled according to the given mode:
    #  > 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
    #  > 'nearest': aaaaaaaa|abcd|dddddddd
    #  > 'reflect': abcddcba|abcd|dcbaabcd
    #  > 'wrap': abcdabcd|abcd|abcdabcd
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../_data/brain/train',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
)

xy_test = test_datagen.flow_from_directory(
    '../_data/brain/test',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
)

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(150, 150, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


start = time.time()
# model.fit(xy_train[:][0], xy_train[:][1])
es = EarlyStopping(monitor='val_loss', patience=24, verbose=2, restore_best_weights=True)
hist = model.fit_generator(xy_train, epochs=24, steps_per_epoch=32, validation_data=xy_test,
                           # validation_steps=4: doesn't work solo w/o validation_data
                           # batch_size is not expected since generator already deals with it.
                           callbacks=[es]
       # steps_per_epoch=the number of images/batch<as specified in generator>    
       )               
end = time.time() - start


acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# val_acc: 0.824999988079071, val_loss: 0.5317312479019165