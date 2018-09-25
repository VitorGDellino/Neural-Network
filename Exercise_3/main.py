"""
Elisa Saltori Trujillo - 8551100
Vitor Giovani Dellinocente - 9277875

SCC 0270 – Redes Neurais
Profa. Dra. Roseli Aparecida Francelin Romero

Exercício 3
"""

from preprocessing import PreProcessing
from rbf import Rbf
from mlp import Mlp
import training

import numpy as np
import math




"""
Main function for mlp and rbf.

"""
def main():
    
    #read dataset and preprocess it
    dataset = PreProcessing("seeds_dataset.txt", separator='\s+')
    dataset.normalize()
    dataset.normalize_class()
    
    #divide dataset into training and test sets
    train, test = training.holdout(0.7, dataset.normalized_dataframe)
    
    nn = Rbf(7, 3)
  

    nn.train(train, eta=0.5, max_iterations=500)

    print("RBF:",training.accuracy(nn, test, 3))

    mm = Mlp(7, 3, 3)

    mm.backpropagation(train.values.tolist(),max_iterations=500)
    print("MLP:",training.accuracy(mm, test, 3))


if __name__ == "__main__":
    main()