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

    
    def define_centers(self, dataset):
        """
        Define centers for network. Each center is the mean of a given class in the dataset

        dataset: input dataset. Class columns should be the last ones to the right.

        """

        #number of classes is given by number of output neurons
        #each class should be a 0 or 1 in the last n columns, one for each class

        self.centers = np.zeros((self.n_neurons_hidden, self.n_neurons_input))
        self.betas = np.zeros((self.n_neurons_hidden))

        n_columns = len(dataset.columns)

        for i in range (0, self.n_neurons_hidden):
            #get all rows in dataset belonging to that class
            #don't get class columns
            class_examples = dataset[dataset[n_columns-i] == 1].iloc[:,:-self.n_neurons_output]

            #find center of each class (mean of all data of that class)
            self.centers[i] = class_examples.mean()

            #sigma is the average of the euclidian distance between the examples and the center
            #beta is 1/(2*sigma*sigma)
            self.betas[i] = np.sum(np.sqrt(np.sum((class_examples-self.centers[i])**2, axis=1)))/len(class_examples.index)
            self.betas[i] = 1/(2*(self.betas[i]**2))


    
    """
    Training function for network
    Parameters:
        dataset: input dataset. Panda dataframe
        eta: learning rate
        threshold: error threshold
    """
    def train(self, dataset, eta=0.3, threshold = 1e-3, momentum=0.5, max_iterations=30000):

        #define centers of given data
        self.define_centers(dataset)

        #numero de iteracoes
        it = 0
        output_momentum = 0

        squaredError = 2*threshold
        while(squaredError > threshold):
            squaredError = 0
            for i in range(len(dataset)):
                
                Xi =  dataset.iloc[i,0:self.n_neurons_input]
                Yi =  dataset.iloc[i,self.n_neurons_input:]
                

                #feed forward
                self.feed_forward(Xi)

                error = np.array(Yi) - np.array(self.output_f_nets)

                squaredError += np.sum(np.power(error, 2))

                aux = eta*np.dot(np.transpose(np.matrix(error)), np.matrix(np.append(self.hidden_outputs, 1)))

                self.output_layer_weights_and_theta += aux

            
            squaredError = squaredError/len(dataset)
            
            #imprime erro a cada 100 iteracoes
            if(it % 100 == 0):
                print("iteration",it," error: ",squaredError)

                
            it +=1

            if(it >= max_iterations):
                print("Maximum number of iterations (", max_iterations,") reached! - Final error:", squaredError)
                return

        print("Error threshold reached in iteration",it,"- Final error:", squaredError)


    # Activation function - radial basis function
    # 
    # Parameters:
    #   beta - distance coefficient
    #   x - input array
    #   c - center array
    def radial_activation_function(self, x, c, beta):

        x = np.array(x)
        return (np.exp(-beta*np.sum((x-c)**2, axis=1)))

    # Sigmoid activation function
    # net -> Sum(xi*wi) + theta(i)
    # theta = bias
   

    # Feed input data to network
    def feed_forward(self, input_data):
        """
        Input data: list containing input values. No class columns allowed.
        """

        #radial neurons
        self.hidden_outputs = self.radial_activation_function(input_data, self.centers, self.betas)


        # Output Layer
        hidden_f_nets = self.hidden_outputs.copy() #hidden f nets da ultima camada
        hidden_f_nets = np.append(hidden_f_nets, 1)
        self.output_nets = []
        self.output_f_nets = []
        output_xi_wi = np.multiply(self.output_layer_weights_and_theta, hidden_f_nets)
        for i in range(output_xi_wi.shape[0]):
            self.output_nets.append(np.sum(output_xi_wi[i]))
            self.output_f_nets.append(self.output_nets[i])


       
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
        print()
        print("OUTPUT NET")
        print(self.output_nets)
        print()
        print("OUTPUT F_NET")
        print(self.output_f_nets)
