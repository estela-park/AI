import numpy as np

x = np.array([range(100), range(301, 401), range(1, 101), range(100), range(401, 501)])
x = np.transpose(x) # (100, 5)
y = np.array([range(711, 811), range(101, 201)])
y = np.transpose(y) # (100, 5)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

# when you can't be bother to name every layer                         
input1 = Input(shape=(5,)) 
hl = Dense(3)(input1) 
hl = Dense(4)(hl)  
hl = Dense(10)(hl)
output1 = Dense(2)(hl) 

model = Model(inputs=input1, outputs=output1)

model.summary()