"""
Elisa Saltori Trujillo - 8551100
Vitor Giovani Dellinocente - 9277875

SCC 0270 – Redes Neurais
Profa. Dra. Roseli Aparecida Francelin Romero

Exercício 2
"""

from mlp import Mlp
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

    #number of hidden neurons = log2(n)
    n_neurons_hiddens = math.log(INPUT_SIZE, 2)

    #generate input: each line of a Id matrix of dimension equal toINPUT_SIZE
    training_input = generate_input()

    #create network with input size of n, log2 n hidden neurons and n output size
    nn = Mlp(INPUT_SIZE, [n_neurons_hiddens, n_neurons_hiddens], INPUT_SIZE, n_hidden_layers=2)

    #train
    nn.backpropagation(training_input, eta=0.5)
    
    #show result for first line
    nn.feed_forward(training_input[0][:INPUT_SIZE])
    nn.show()


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