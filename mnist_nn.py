import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

import matplotlib.pyplot as plt
image_index = 7777
print(y_train[image_index])
plt.imshow(x_train[image_index], cmap='Greys')

x_train.shape

#Reshaping the array to a 4-dimensional so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalizing the RGB codes by dividing it to the max RGB values
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('NUmber of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

#Import the required Keras modules containing model and layers
from keras.model import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
#Creating a sequential model and adding the layers
model = Sequential()
model.add(Conv2d(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten()) #Flatten the 2D arrays for the fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))


model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=10)

model.evauate(x_test, y_test)
