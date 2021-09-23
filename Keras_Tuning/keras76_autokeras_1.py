### autokeras.com > Documentation ###

# autokeras do the preprocessing and hyper-parameter tunings and likes.
# ImageClassifier/Regressor
# StructuredDataClassifier/Regressor

import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
model = ak.ImageClassifier(overwrite=True, max_trials=2)
# max_trial = how many times the machine needs training
model.fit(x_train, y_train, epochs=5)
pred = model.predict(x_test[:3])
evaluation = model.evaluate(x_test, y_test)