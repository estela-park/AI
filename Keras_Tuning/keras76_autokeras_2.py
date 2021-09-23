import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
model = ak.ImageClassifier(overwrite=True, max_trials=1)
# ImageClassifier doesn't take 1D input,
# that is, after scaling 1D vector should be de-flattened
model.fit(x_train, y_train, epochs=1)
pred = model.predict(x_test[:3])
# pred will be in the shape the user inputs its label values
evaluation = model.evaluate(x_test, y_test)

model1 = model.export_model()
model1.summary()


# Model: "functional_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param 
# =================================================================
# input_1 (InputLayer)         [(None, 28, 28)]          0 

# _________________________________________________________________
# cast_to_float32 (CastToFloat (None, 28, 28)            0 

# _________________________________________________________________
# expand_last_dim (ExpandLastD (None, 28, 28, 1)         0 

# _________________________________________________________________
# normalization (Normalization (None, 28, 28, 1)         3 

# _________________________________________________________________
# conv2d (Conv2D)              (None, 26, 26, 32)        320
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0
# _________________________________________________________________
# dropout (Dropout)            (None, 12, 12, 64)        0
# _________________________________________________________________
# flatten (Flatten)            (None, 9216)              0
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 9216)              0
# _________________________________________________________________
# dense (Dense)                (None, 10)                92170
# _________________________________________________________________
# classification_head_1 (Softm (None, 10)                0
# =================================================================
# Total params: 110,989