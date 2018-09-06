from mlp import Mlp
import numpy as np
import math


INPUT_SIZE = 4

def main():
    
    n_neurons_hiddens = math.log(INPUT_SIZE, 2)

    training_input = generate_input()

    
    nn = Mlp(INPUT_SIZE, n_neurons_hiddens, INPUT_SIZE)
    #nn.show()
    nn.backpropagation(training_input)
    print(training_input[0])
    nn.feed_forward(training_input[0][:INPUT_SIZE])
    nn.show()
    """
    nn = Mlp(2, 2, 1)
    nn.backpropagation([[1,0,1],[0,1,1],[0,0,0],[1,1,0]])
    nn.feed_forward([1,1])
    nn.show()
    nn.feed_forward([1,0])
    nn.show()
    nn.feed_forward([0,1])
    nn.show()
    nn.feed_forward([0,0])
    nn.show()
    """

    

def generate_input():
    
    training_input = []
    g = [0]*INPUT_SIZE
    n = int(math.sqrt(INPUT_SIZE)+1)
    g[::n]=map(lambda x: 1, g[::n])
    g = g*2
    training_input.append(g)

    return training_input


#create neural network
#generate input
#feed it



if __name__ == "__main__":
    main()