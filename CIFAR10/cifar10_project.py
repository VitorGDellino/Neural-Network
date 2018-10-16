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
from keras import optimizers

# Helper libraries
import numpy as np


import matplotlib.pyplot as plt
import random
import imageio
import os
import glob


test_directory = "test_images"
label_names = ["airplane", "automobile", "bird", "cat",
"deer", "dog", "frog", "horse", "ship", "truck"]
model_name = 'keras_cifar10_trained_model.h5'


# input image dimensions
img_rows, img_cols = 32, 32

# input number of color channel
n_channels = 3


def main():

  #train model
  #model = train()
  model = load_model()

  #test on saved images
  test_on_saved_images(model)


def train():

  save_dir = os.path.join(os.getcwd(), 'saved_models')
  

  batch_size = 32
  epochs = 20

  #loading data
  cifar10 = keras.datasets.cifar10 
  (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

  #normalizing data to [0,1]
  train_images = train_images / 255.0
  test_images = test_images / 255.0

  
  #reshape data for use in network - from 3D to 4D (add channels dimension)
  if K.image_data_format() == 'channels_first':
      train_images = train_images.reshape(train_images.shape[0], n_channels, img_rows, img_cols)
      test_images = test_images.reshape(test_images.shape[0], n_channels, img_rows, img_cols)
      input_shape = (n_channels, img_rows, img_cols)
  else:
      train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, n_channels)
      test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, n_channels)
      input_shape = (img_rows, img_cols, n_channels)


  #building model
  model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])

  model.compile(optimizer=keras.optimizers.Adam(lr=0.0001, decay=1e-6), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  #train model
  #model.fit(train_images, train_labels, epochs=10)
  model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, shuffle=True)

  #test accuracy
  test_loss, test_acc = model.evaluate(test_images, test_labels)

  print('Test accuracy:', test_acc)

  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
  model_path = os.path.join(save_dir, model_name)
  model.save(model_path)
  print('Saved trained model at %s ' % model_path)

  return model


def load_model():

  filepath = os.path.join(os.getcwd(), 'saved_models', model_name)
  model = keras.models.load_model(filepath)
  return model

def test_on_saved_images(model):
  """
  Get images from the test directory and use model to predict output
  """

  image_list = []
  labels = []

  #read all images in test directory
  path = os.path.join(test_directory, "*_.jpg")

  for filename in glob.glob(path): 
      im=imageio.imread(filename)
      labels.append(filename)
    
      #reverse image
      #im = 255 - im
      
      #normalize image
      im = im/255.0

      #append to list
      image_list.append(im[:,:,:])

  images = np.array(image_list)

    #reshape data for use in network - from 3D to 4D (add channels dimension)
  if K.image_data_format() == 'channels_first':
      images = images.reshape(images.shape[0], n_channels, img_rows, img_cols)
  else:
      images = images.reshape(images.shape[0], img_rows, img_cols, n_channels)

  #predictions for each one:
  predictions = model.predict(images)

  for i in range(0, predictions.shape[0]):
    print(str(labels[i])+":", predictions[i])
    print("Result:", np.argmax(predictions[i]), "("+label_names[np.argmax(predictions[i])]+")")


if __name__ == "__main__":
    main()

