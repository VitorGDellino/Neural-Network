"""
Elisa Saltori Trujillo - 8551100
Vitor Giovani Dellinocente - 

SCC 0270 – Redes Neurais
Profa. Dra. Roseli Aparecida Francelin Romero

Exercício 1

"""

import numpy as np
import glob
import os

EXAMPLE_SIZE = 5
WEIGHTS_FILE = 'weights.txt'
NUMBER_OF_EXAMPLES = 12
MAX_ITERATIONS = 100
LEARNING_RATE = 0.5

#activation function: 1 or -1
def step(x):
    if (x > 0):
        return 1
    else:
        return -1

def adaline():
    """
    Adaline implementation to classify 5x5 matrixes as
    A (-1) and inverted As (+1).

    -Read weights from specified file (else initiates weights with 0s)
    -Read examples from specified path
        Format:
            classification (1 or -1) in first line followed by matrix of
        input values

            Ex:
                -1
                -1 -1 +1 -1 -1
                -1 +1 -1 +1 -1
                -1 +1 +1 +1 -1
                +1 -1 -1 -1 +1
                +1 -1 -1 -1 +1
    -Run training until weights stabilize or the maximum number of iterations
is reached
    -Print final weights and save them to file

    """

    expected_results = [] #the correct classification of each example
    examples = [] #array containing the input examples for training
    weights = np.zeros((EXAMPLE_SIZE*EXAMPLE_SIZE+1))

    #initialize weights
    #looks for weights in file called weights.txt
    #else weights are kept as zero
    try:
        weights = np.loadtxt(WEIGHTS_FILE)
    except IOError:
        print(WEIGHTS_FILE, "not found. Initial weights will be set to 0.")

    #read training examples from files
    path = '../training' #filepath for training files

    for filename in glob.glob(os.path.join(path, '*.txt')):

        with open(filename) as f:
            expected_results.append(int(next(f)))

        examples.append(np.reshape(np.loadtxt(filename, skiprows=1), newshape=(1, EXAMPLE_SIZE*EXAMPLE_SIZE)))

    
    
    ##main loop for training
    for i in range(0, MAX_ITERATIONS):

        changed = False

        #for each example
        for j in range(0, NUMBER_OF_EXAMPLES):
            
            #output
            output = weights[0] + np.sum(np.multiply(examples[j], weights[1::]))

            #activation function: hard limiter
            output = step(output)
            
            #error for the example
            error = expected_results[j] - output
            

            #training should continue to next cycle if error is different from 0
            if(error != 0):
                changed = True

            #updates weights
            weights[0] += LEARNING_RATE*error #bias
            weights[1::] =  weights[1::]+ LEARNING_RATE*error*examples[j]
        

        #if all examples for the loop were classified correctly, no weights were changed
        #end training
        if(changed == False):
            break

    ##prints weights and save them to file
    print("Final weights:")
    print(weights)

    np.savetxt(WEIGHTS_FILE, weights)

    
if __name__ == "__main__":
    adaline()