from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

url_training = 'https://pjreddie.com/media/files/mnist_train.csv'
url_testing = 'https://pjreddie.com/media/files/mnist_test.csv'

df_train = pd.read_csv(url_training, header=None)
df_test = pd.read_csv(url_testing, header=None)

# combine training and testing 
data = np.concatenate((df_train, df_test))

# Check if data was correctly combined
#print('data size:', data.shape)
#print('data concatenate:', data)


total_avg = 0

for i in range(100):
  np.random.shuffle(data)


  x = data[:, 1:]
  y = data[:, 0]


  #k-10 fold

  x_train = x[:63000, :]
#  print('x_train_shape', x_train.shape)
  y_train = y[:63000]
#  print('y_train_shape', y_train.shape)
  x_test = x[63000:, :]
  y_test = y[63000:]

# Reshape x so instead of having 748 some pixels in a single x row we have a 28 x 28 image
# We also change the type from the origial int value to a float type so we can have ...
# more accurate numbers when we normalize when we divide by 255 from the greyscale it uses
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255

# convert y using one-hot method for mutliclass 
  y_train = np_utils.to_categorical(y_train, 10)
  y_test = np_utils.to_categorical(y_test, 10)


  model = Sequential()
  model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Dropout(0.2))
  model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Dropout(0.2))
  model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(64, activation='relu'))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(10, activation='softmax'))



  model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

  history = model.fit(x_train, y_train, epochs=30, batch_size=400)

  y_loss = history.history['accuracy']


  print(model.evaluate(x_test, y_test)[1])
  total_avg += model.evaluate(x_test, y_test)[1]

print('total_avg: ', total_avg/100 )
