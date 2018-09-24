"""
Elisa Saltori Trujillo - 8551100
Vitor Giovani Dellinocente - 9277875

SCC 0270 – Redes Neurais
Profa. Dra. Roseli Aparecida Francelin Romero

Exercício 3
"""
import numpy as np

import math


class Rbf:
    # Creating RBF network
    #
    # For this exercise, the number of hidden neurons is the same as n_neurons_output (the number of classes)
    # n_neurons_input -> number of neurons in input layer
    # n_neurons_output -> number of neurons in output layer
    # output_layer_weights_and_theta -> Matriz contendo os valores predeterminados com os pesos e o bias da camada de saida, sendo que o bias é a ultima coluna da matriz
    def __init__(self, n_neurons_input, n_neurons_output, output_layer_weights_and_theta = None):
        # Defining the number of neurons per layer
        self.n_neurons_input = math.ceil(n_neurons_input)
        self.n_neurons_output = math.ceil(n_neurons_output)
        self.n_neurons_hidden = self.n_neurons_output

        # Initializing output weights
        self.init_output_layer_weight_and_theta(output_layer_weights_and_theta)
    
    # Initializing weights of output layer
    # output_layer_weights_and_theta -> Matrix containing predetermined weights and bias of the output layer, the bias being in the last column of the matrix
    def init_output_layer_weight_and_theta(self, output_layer_weights_and_theta):
        if not output_layer_weights_and_theta:
            self.output_layer_weights_and_theta = np.random.uniform(-0.5, 0.5, (self.n_neurons_output, self.n_neurons_hidden + 1))
        else:
            self.output_layer_weights_and_theta = output_layer_weights_and_theta

    # Activation function - radial basis function
    # 
    # Parameters:
    #   beta - distance coefficient
    #   x - input array
    #   c - center array
    def activation_function(self, x, c, beta):
        #UPDATE ME SO I CAN DO ALL THE OUTPUTS AT THE SAME TIME?
        return (np.exp(-beta*np.sqrt(np.sum((x-c)**2))))

    
    def define_centers(aelf, dataset):
        """
        Define centers for network. Each center is the mean of a given class in the dataset

        """

        #number of classes is given by number of output neurons
        #each class should be a 0 or 1 in the last n columns, one for each class

        self.centers = np.zeros((self.n_neurons_hidden, self.n_neurons_input))
        self.betas = np.zeros((self.n_neurons_hidden))

        n_columns = len(dataset.columns)

        for i in range (0, self.n_neurons_hidden):
            #get all rows in dataset belonging to that class
            class_examples = dataset[dataset[n_columns-i] == 1]

            #find center of each class (mean of all data of that class)
            self.centers[i] = class_examples.mean()

            #beta is the average of the euclidian distance between the examples and the center
            self.betas[i] = np.sum(np.sqrt(np.sum((class_examples-self.centers[i])**2, axis=1)))/len(class_examples.index)

    
    """
    Função de treino utilizando o algoritmo backpropagation
    Parametros:
        dataset: dataset de entrada, como descrito em main.py
        eta: constante de treinamento
        threshold: limite de tolerancia do erro

    """
    def train(self, dataset, eta=0.3, threshold = 1e-3):

        #define centers of given data
        self.define_centers(dataset)

        #numero de iteracoes
        it = 0

        squaredError = 2*threshold
        while(squaredError > threshold):
            squaredError = 0
            for i in range(len(dataset)):
                Xi =  dataset[i] [0:self.n_neurons_input]
                Yi =  dataset[i][self.n_neurons_input:len(dataset[0])]

                
                #feed forward
                #DO FEED FORWARD HERE 


                error = np.array(Yi) - np.array(self.output_f_nets)

                squaredError += np.sum(np.power(error, 2))

                #UPDATE WEIGHTS HERE!!!
            
            squaredError = squaredError/len(dataset)
            
            #imprime erro a cada 100 iteracoes
            if(it % 100 == 0):
                print("iteration",it," error: ",squaredError)
            it +=1

    # Realiza o forward da rede neural
    def feed_forward(self, input_data):
        pass

    # Mostra a rede neural
    def show(self):
        print("INPUT " + str(self.n_neurons_input))
        print("HIDDEN " + str(self.n_neurons_hidden))
        print("OUTPUT " + str(self.n_neurons_output))

        print("--------HIDDEN LAYER--------")
        print("----------------------------")
        print()
        print("--------OUTPUT LAYER--------")
        print(self.output_layer_weights_and_theta)
        print("----------------------------")
        print()
        print("HIDDEN NET")
        print(self.hidden_nets)
        print()
        print("HIDDEN F_NET")
        print(self.hidden_f_nets)
        print()
        print("OUTPUT NET")
        print(self.output_nets)
        print()
        print("OUTPUT F_NET")
        print(self.output_f_nets)
