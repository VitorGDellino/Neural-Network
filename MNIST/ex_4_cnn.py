# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import imageio
import os
import glob


test_directory = "test_images"

# input image dimensions
img_rows, img_cols = 28, 28


def train():

  #loading data
  mnist = keras.datasets.mnist
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

  #normalizing data to [0,1]
  train_images = train_images / 255.0
  test_images = test_images / 255.0

  
  #reshape data for use in network - from 3D to 4D (add channels dimension)
  if K.image_data_format() == 'channels_first':
      train_images = train_images.reshape(train_images.shape[0], 1, img_rows, img_cols)
      test_images = test_images.reshape(test_images.shape[0], 1, img_rows, img_cols)
      input_shape = (1, img_rows, img_cols)
  else:
      train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
      test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
      input_shape = (img_rows, img_cols, 1)


  #building model
  model = keras.Sequential([
    keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    keras.layers.Conv2D(16, (5, 5), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])

  model.compile(optimizer=tf.train.AdamOptimizer(), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  #train model
  model.fit(train_images, train_labels, epochs=1)

  #test accuracy
  test_loss, test_acc = model.evaluate(test_images, test_labels)

  print('Test accuracy:', test_acc)

  #predictions = model.predict(test_images)

  return model


def main():

  #train model
  model = train()


  #check for directory test_images
  #if it doesnt exist, create it and fill it with some test_images
  if(not os.path.exists(test_directory)):
    save_as_images()
  
  #test on saved images
  test_on_saved_images(model)



def test_on_saved_images(model):

  image_list = []
  labels = []

  #read all images in test directory
  path = os.path.join(test_directory, "*.png")

  for filename in glob.glob(path): 
      im=imageio.imread(filename)
      labels.append(filename)
      image_list.append(im)

  images = np.array(image_list)
  print(images.shape)

    #reshape data for use in network - from 3D to 4D (add channels dimension)
  if K.image_data_format() == 'channels_first':
      images = images.reshape(images.shape[0], 1, img_rows, img_cols)
  else:
      images = images.reshape(images.shape[0], img_rows, img_cols, 1)

  #predictions for each one:
  predictions = model.predict(images)

  for i in range(0, predictions.shape[0]):
    print(str(labels[i])+":", predictions[i])


  



def save_as_images():
  """
  Select 4 random images from set pertaining to different labels and save them as png
  """

  #loading data
  mnist = keras.datasets.mnist
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

  #create directory for files
  os.makedirs(test_directory, exist_ok=True)

  count = 0
  number_user = [0]*10 #used for checking if a label has already been used

  while(count < 4):
    #get a random image from test set
    index = random.randint(0, test_images.shape[0]-1)

    #check if class has already been selected
    if(number_user[test_labels[index]]==0):

      #mark label as used
      number_user[test_labels[index]] = 1

      path = os.path.join(test_directory, str(test_labels[index])+".png")
      #save image
      imageio.imwrite(path, test_images[index], "png")
      count += 1

  


if __name__ == "__main__":
    main()

