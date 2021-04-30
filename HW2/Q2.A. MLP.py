#write your code here

import keras
from keras.layers import *
from keras.optimizers import *
from keras.models import Sequential

import numpy as np
import matplotlib.pyplot as plt


# create inputs for sin(x)
x = np.arange(-150, 150).reshape(-1,1) / 50
y = np.sin(x)

# create MLP
model = Sequential()

model.add(Dense(30, input_shape=(1,)))
model.add(Activation('sigmoid'))
model.add(Dense(20))
model.add(Activation('sigmoid'))
model.add(Dense(1))
print(model.summary())
model.compile(loss='mean_squared_error', optimizer='SGD')

# traing MLP for sin(x)
result = model.fit(x, y, epochs=1000, batch_size=4, verbose= 0)

# predict sin(x) that trained before
predictions = model.predict(x)

# plot prediction and sin(x) together 
plt.plot(x,predictions)
plt.plot(x,y)
plt.xlabel('MLP')
plt.legend(["predict", "sinx"])
plt.show()





