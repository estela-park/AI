# AutoModel resembles funtional model, allowing user more freedom
import autokeras as ak
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

inputs = ak.ImageInput()
hidden = ak.ImageBlock(block_type='resnet', normalize=True, augment=False)(inputs)
outputs = ak.ClassificationHead()(hidden)

model = ak.AutoModel(inputs=inputs, outputs=outputs, overwrite=True, max_trials=1)
model.fit(x_train, y_train, epochs=1)
pred = model.predict(x_test[:3])
evaluation = model.evaluate(x_test, y_test)

model1 = model.export_model()
model1.summary()