# Q4_graded
# Do not change the above line.

# This cell is for your imports.
import keras
from keras.layers import *
from keras.optimizers import *
from keras.models import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Q4_graded
# Do not change the above line.

# This cell is for your codes.

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape inputs and labels to fit in mlp
x_train = x_train.reshape(60000,784).astype('float32')
x_test = X.reshape(10000,784).astype('float32')
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# normilize 
x_train = x_train / 255.0
x_test = x_test / 255.0
# create model and layeres
model = Sequential()
model.add(Dense(85, input_shape=(784,), activation='relu'))
# model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile and optimize model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

# train the model 
result = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_test, y_test)) 

##plotting the results 
# accuracy
plt.plot(result.history['acc'])
plt.plot(result.history['val_acc'])
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy in each iteration')
plt.legend(["training", "test"])
plt.show()

#loss
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss in each iteration')
plt.legend(["training", "test"])
plt.show()

# Q4_graded
# Do not change the above line.

# This cell is for your codes.

