from mlp import Mlp
from preprocessing import PreProcessing
import training
from rbf import Rbf

import numpy as np
import math



def seed_test():
    # Carregando e Normalizando os dados da base de vinhos
    dataset = PreProcessing("seeds_dataset.txt", separator='\s+')
    dataset.normalize()
    dataset.normalize_class()

    # Atributos a serem variados nos testes
    n_layers = [1,2]
    hidden_layer = [3, [6,6]]
    momentums = [0.3, 0.5]
    max_iterations = [100,250,500]
    etas = [0.3, 0.5]
    ps = [0.7, 0.9]

    rbf_accuracy = 0
    mlp_accuracy = 0
    tests = 0

    # Teste
    for layer in n_layers:
        for momentum in momentums:
            for eta in etas:
                for max_iteration in max_iterations:
                    for p in ps:
                        tests +=1

                        print("Test number", tests)

                        train, test = training.holdout(p, dataset.normalized_dataframe)
                        print("INPUT NEURONS = 7 HIDDEN NEURONS = "+ str(int(6/layer)) +" OUTPUT NEURONS = 3 HIDDEN LAYER = " + str(layer) + " ETA = " + str(eta) + " MAX ITERATIONS = " + str(max_iteration) + " MOMENTUM = " + str(momentum) + " P = " + str(p))
                        print()
                        print("RBF")

                        nn = Rbf(7, 3)

                        nn.train(train, eta=0.5, max_iterations=max_iteration) 
                        ac = training.accuracy(nn, test, 3)
                        rbf_accuracy += ac
                        print("ACCURACY =", ac)

                        print()
                        print("MLP")
                        example = test.values.tolist()
                      
                        mm = Mlp(7, hidden_layer[layer - 1], 3, n_hidden_layers=layer)
                        mm.backpropagation(train.values.tolist(), eta=eta, max_iterations=max_iteration)
                        ac = training.accuracy(mm, test, n_classes=3)
                        mlp_accuracy += ac
                        print("ACCURACY =", ac)
                        print()

                        print("Rbf:")
                        nn.feed_forward(example[15][:(-1*3)])
                        print(example[15])
                        print("Result 1")
                        nn.show_class()
                        print()

                        print("Mlp")
                        print(example[15])
                        nn.feed_forward(example[15][:(-1*3)])
                        print("Result 2")
                        mm.show_class()
                        print()
                        print("******************************************************//******************************************************")
                        print()

    print(tests," tests executed. Rbf accuracy:", rbf_accuracy/tests," Mlp accuracy:", mlp_accuracy/tests)


                        

def main():
    seed_test()

    
    
  
    

if __name__ == "__main__":
    main()