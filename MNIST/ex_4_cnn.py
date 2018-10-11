"""
Elisa Saltori Trujillo - 8551100
Vitor Giovani Dellinocente - 9277875

SCC 0270 – Redes Neurais
Profa. Dra. Roseli Aparecida Francelin Romero

Exercício 4
"""

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


def main():

  #train model
  model = train()

  #test on saved images
  test_on_saved_images(model)


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
    keras.layers.Conv2D(8, kernel_size=(8, 8), activation='relu', input_shape=input_shape),
    keras.layers.Conv2D(16, (4, 4), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])

  model.compile(optimizer=tf.train.AdamOptimizer(), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  #train model
  model.fit(train_images, train_labels, epochs=2)

  #test accuracy
  test_loss, test_acc = model.evaluate(test_images, test_labels)

  print('Test accuracy:', test_acc)

  return model


def test_on_saved_images(model):
  """
  Get images from the test directory and use model to predict output
  """

  image_list = []
  labels = []

  #read all images in test directory
  path = os.path.join(test_directory, "*.png")

  for filename in glob.glob(path): 
      im=imageio.imread(filename)
      labels.append(filename)
      print(im)
      #reverse image
      im = 255 - im
      image_list.append(im)

  images = np.array(image_list)

    #reshape data for use in network - from 3D to 4D (add channels dimension)
  if K.image_data_format() == 'channels_first':
      images = images.reshape(images.shape[0], 1, img_rows, img_cols)
  else:
      images = images.reshape(images.shape[0], img_rows, img_cols, 1)

  #predictions for each one:
  predictions = model.predict(images)

  for i in range(0, predictions.shape[0]):
    print(str(labels[i])+":", predictions[i])


if __name__ == "__main__":
    main()

