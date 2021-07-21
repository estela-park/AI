from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model

datasets = load_diabetes()

x = datasets.data   # (442, 10)
y = datasets.target # (442, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=78)

model = load_model('./_save/modelcheckpoint/keras47_MCP_model.h5')
model.load_weights('./_save/modelcheckpoint/keras47_MCP.hdf5')
loss = model.evaluate(x_test, y_test)
predict = model.predict(x_test)

r2 = r2_score(y_test, predict)

print('loss:', loss, 'actual data:', y_test, 'machine predicted:', predict)
print('accuracy:', r2)

'''
-saved at random_state=78
loss: [2748.679931640625, 44.2568244934082] accuracy: 0.5462202791204794  
-loaded at random_state=78
loss: [2748.679931640625, 44.2568244934082] accuracy: 0.5462202791204794
-with weight saved with checkpoint at random_state=78
loss: [2925.89892578125, 45.4714241027832] accuracy: 0.5169631500117884
'''