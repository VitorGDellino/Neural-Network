"""
Elisa Saltori Trujillo - 8551100
Vitor Giovani Dellinocente - 9277875

SCC 0270 – Redes Neurais
Profa. Dra. Roseli Aparecida Francelin Romero

Projeto 2

This file loads a model trained on the CIFAR10 dataset
and classifies 10 sample images.

"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import optimizers

# Helper libraries
import numpy as np


import matplotlib.pyplot as plt
import imageio
import os
import glob
import re


test_directory = "test_images"
label_names = ["airplane", "automobile", "bird", "cat",
"deer", "dog", "frog", "horse", "ship", "truck"]
model_name = 'keras_cifar10_trained_model.h5'

# input image dimensions
img_rows, img_cols = 32, 32

# input number of color channel
n_channels = 3


def main():

  #load model from disk
  model = load_model()

  #test on saved images
  test_on_saved_images(model)


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

  #set to 4 decimal places for readability
  np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})


  print("Output:")
  print()

  acc = 0

  for i in range(0, predictions.shape[0]):


    
    #check if label is correct (label comes from filename)
    correct = False
    p = re.compile(label_names[np.argmax(predictions[i])])
    m = p.search(labels[i])
    if(m != None):
      correct = True

    print(str(labels[i])+":", predictions[i])
    print("Result:", np.argmax(predictions[i]), "("+label_names[np.argmax(predictions[i])]+")")

    if(correct == True):
      acc += 1
      print("Correct")
    else:
      print("Incorrect")
    print()

  print("Accuracy: "+str(acc)+"/"+str(predictions.shape[0])+" ("+str(acc/predictions.shape[0]*100)+"%)")


if __name__ == "__main__":
    main()

