import autokeras as ak
import pandas as pd
from sklearn.datasets import load_boston

datasets = load_boston()
x = datasets.data
y = datasets.target

model = ak.StructuredDataRegressor(overwrite=True, max_trials=1)
model.fit(x, y, epochs=1, validation_split=0.15)

print(model.evaluate(x[:50], y[:50]))

model_temp = model.export_model()
model_temp.summary()

# Model: "functional_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param # 

# =================================================================
# input_1 (InputLayer)         [(None, 13)]              0       

# _________________________________________________________________
# multi_category_encoding (Mul (None, 13)                0       

# _________________________________________________________________
# normalization (Normalization (None, 13)                27      

# _________________________________________________________________
# dense (Dense)                (None, 32)                448     

# _________________________________________________________________
# re_lu (ReLU)                 (None, 32)                0       

# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                1056    

# _________________________________________________________________
# re_lu_1 (ReLU)               (None, 32)                0       

# _________________________________________________________________
# regression_head_1 (Dense)    (None, 1)                 33      

# =================================================================
# Total params: 1,564