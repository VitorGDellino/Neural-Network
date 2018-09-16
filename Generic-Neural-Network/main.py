"""
Elisa Saltori Trujillo - 8551100
Vitor Giovani Dellinocente - 9277875

SCC 0270 – Redes Neurais
Profa. Dra. Roseli Aparecida Francelin Romero

Exercício 2
"""



"""
TODO
-Read some parameters from input
-Make function which reads input and separates it in training and testing
-specific functions for wine and default features
-For wine, round numbers to nearest class
-accuracy for wine and erro quadratico medio

"""

from mlp import Mlp
from preprocessing import PreProcessing
import training

import numpy as np
import math


INPUT_SIZE = 10


"""
Main function for mlp.

Input should be given as an array of arrays, 
with each input being a single array containing first the input values
and then the expect output values

Example:
[[0 1 1 2], [0 0 0 0]]

Would have two lines of inputs.
For the first input, (0 1) would be the input values
and (1 2) the expected output.

"""
def main():
    
    dataset = PreProcessing("wine_dataset.txt")
    dataset.normalize(ignore_first_column=True)
    dataset.switch_first_last_column()
    dataset.normalize_class()

    train, test = training.holdout(0.7, dataset.normalized_dataframe)
    

    nn = Mlp(13, 10, 3, n_hidden_layers=1)
    nn.backpropagation(train.values.tolist(), eta=0.5)

    print(training.accuracy(nn, test, n_classes=3))

    Input1 = test.iloc[[5]].values.tolist()
    nn.feed_forward(Input1[:(-1*3)])

    print()
    print("Input")
    print(Input1)
    print("Class")
    nn.show_class()
    print()

    Input2 = test.iloc[[40]].values.tolist()
    nn.feed_forward(Input2[:(-1*3)])
    
    print()
    print("Input")
    print(Input2)
    print("Class")
    nn.show_class()
    print()

    Input3 = test.iloc[[30]].values.tolist()
    nn.feed_forward(Input3[:(-1*3)])
    
    print()
    print("Input")
    print(Input3)
    print("Class")
    nn.show_class()
    print()
    
    #dataset = PreProcessing("default_features_1059_tracks.txt")
    #dataset.normalize(ignore_first_column=False)
    #turn dataset into list!! OK
    #dataset.normalized_dataframe.values.tolist()
    #turn output classes into normalized representation OK

    #then divide it into training and testing sets OK

    #DO THIS
    #then, chaning parameters:
        #train neural network
        #get accuracy

    #repeat for wine
    #train, test = training.holdout(0.7, dataset.normalized_dataframe)
    #print(len(dataset.normalized_dataframe))
    #print(len(train))
    #print(len(test))
    #print(train)
    #print(test)

    #nn = Mlp(68, [10,10], 2, n_hidden_layers=2)
    #print(dataset.normalized_dataframe.values.tolist())
    #train
    #nn.backpropagation(train.values.tolist(), eta=0.5, max_iterations=200)

    #print(training.accuracy(nn, test, n_classes=3))

    #print(training.squared_error(nn, test, n_classes=2))
    #show result for first line
    
    #nn.show()
    

"""
Generate input for the multilayer perceptron

A INPUT_SIZExINPUT_SIZE identity matrix where each line is an input array
"""
def generate_input():

    training_input = [[0]*INPUT_SIZE*2 for x in range(INPUT_SIZE)]

    for i in range (0, INPUT_SIZE):
        training_input[i][i]=1
        training_input[i][i+INPUT_SIZE]=1

    return (training_input)






if __name__ == "__main__":
    main()