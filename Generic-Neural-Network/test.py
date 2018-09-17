from mlp import Mlp
from preprocessing import PreProcessing
import training

import numpy as np
import math


INPUT_SIZE = 10

def wine_test():
    # Carregando e Normalizando os dados
    dataset = PreProcessing("wine_dataset.txt")
    dataset.normalize(ignore_first_column=True)
    dataset.switch_first_last_column()
    dataset.normalize_class()

    n_layers = [1,2]
    hidden_layer = [10, [5,5]]
    momentums = [0.3, 0.5, 0.7]
    max_iterations = [100,250,500]
    etas = [0.3, 0.5, 0.7]
    ps = [0.5, 0.7, 0.9]

    for layer in n_layers:
        for momentum in momentums:
            for eta in etas:
                for max_iteration in max_iterations:
                    for p in ps:
                        train, test = training.holdout(p, dataset.normalized_dataframe)
                        example = test.values.tolist()
                        print("INPUT NEURONS = 13 HIDDEN NEURONS = "+ str(int(10/layer)) +" OUTPUT NEURONS = 3 HIDDEN LAYER = " + str(layer) + " ETA = " + str(eta) + " MAX ITERATIONS = " + str(max_iteration) + " MOMENTUM = " + str(momentum) + " P = " + str(p))
                        print()
                        nn = Mlp(13, hidden_layer[layer - 1], 3, n_hidden_layers=layer)
                        nn.backpropagation(train.values.tolist(), eta=eta, max_iterations=max_iteration)
                        print("ACCURACY =", training.accuracy(nn, test, n_classes=3))
                        print()

                        print("Input 1")
                        nn.feed_forward(example[0][:(-1*3)])
                        print(example[0])
                        print("Result 1")
                        nn.show_class()
                        print()

                        print("Input 2")
                        print(example[15])
                        nn.feed_forward(example[15][:(-1*3)])
                        print("Result 2")
                        nn.show_class()
                        print()
                        print("******************************************************//******************************************************")
                        print()
                        


def music_test():
    dataset = PreProcessing("default_features_1059_tracks.txt")
    dataset.normalize(ignore_first_column=False)

    n_layers = [1,2]
    hidden_layer = [20, [10,10]]
    momentums = [0.3, 0.5, 0.7]
    max_iterations = [100,250,500]
    etas = [0.3, 0.5, 0.7]
    ps = [0.5, 0.7, 0.9]

    for layer in n_layers:
        for momentum in momentums:
            for eta in etas:
                for max_iteration in max_iterations:
                    for p in ps:
                        train, test = training.holdout(p, dataset.normalized_dataframe)
                        example = test.values.tolist()
                        print("INPUT NEURONS = 68 HIDDEN NEURONS = "+ str(int(10/layer)) +" OUTPUT NEURONS = 2 HIDDEN LAYER = " + str(layer) + " ETA = " + str(eta) + " MAX ITERATIONS = " + str(max_iteration) + " MOMENTUM = " + str(momentum) + " P = " + str(p))
                        print()
                        nn = Mlp(68, hidden_layer[layer - 1], 2, n_hidden_layers=layer)
                        nn.backpropagation(train.values.tolist(), eta=eta, max_iterations=max_iteration)
                        print("ACCURACY =", training.accuracy(nn, test, n_classes=2))
                        print()

                        print("Input 1")
                        nn.feed_forward(example[0][:(-1*2)])
                        print(example[0])
                        print("Result 1")
                        nn.show_class()
                        print()

                        print("Input 2")
                        print(example[15])
                        nn.feed_forward(example[15][:(-1*2)])
                        print("Result 2")
                        nn.show_class()
                        print()
                        print("******************************************************//******************************************************")
                        print()

def main():
    print("Enter 1 to use wine dataset or 2 to use music dataset")
    opt = int(input())
    if opt == 1:
        print("Using Wine DATASET")
        wine_test()
    elif opt == 2:
        print("Using Music DATASET")
        music_test()

    
    
  
    

if __name__ == "__main__":
    main()