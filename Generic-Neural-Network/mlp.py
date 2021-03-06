"""
Elisa Saltori Trujillo - 8551100
Vitor Giovani Dellinocente - 9277875

SCC 0270 – Redes Neurais
Profa. Dra. Roseli Aparecida Francelin Romero

Exercício 2
"""
import numpy as np

import math


class Mlp:
    # Criando a Rede MLP
    # n_neurons_input -> Numero de neuronios na camada de entrada, ou seja o tamanho da entrada
    # n_neurons_hiddens -> Numero de neuronios nas camadas escondidas (para mais de uma camada, usar um array)
    # n_neurons_output -> Numero de neuronios na camada de saida, ou seja o tamanho da saida
    # Os parâmetros abaixo nao são necessários, pois eles sao gerados aletaoriamente usando valores entre -0.5 e 0.5 (Valores positivos e negativos ajudam na convergencia)
    # hidden_layer_weights_and_theta -> Matriz contendo os valores predeterminados com os pesos e o bias das camadas escondidas, sendo que o bias é a ultima coluna da matriz
    # hidden_layer_weights_and_theta -> Matriz contendo os valores predeterminados com os pesos e o bias da camada de saida, sendo que o bias é a ultima coluna da matriz
    def __init__(self, n_neurons_input, n_neurons_hiddens, n_neurons_output, n_hidden_layers=1, hidden_layer_weights_and_theta = None, output_layer_weights_and_theta = None):
        # Setando o numero de neuronios em cada camada
        self.n_neurons_input = math.ceil(n_neurons_input)
        self.n_neurons_output = math.ceil(n_neurons_output)
        self.n_hidden_layers = math.ceil(n_hidden_layers)
        if(n_hidden_layers==1):
            self.n_neurons_hiddens = [math.ceil(n_neurons_hiddens)]
        else:
            self.n_neurons_hiddens= np.ceil(n_neurons_hiddens).astype('int')   

        # Setando os pesos e o bias nas camadas escondida e de saida, respectivamente)
        self.init_hidden_layer_weight_and_theta(hidden_layer_weights_and_theta)
        self.init_output_layer_weight_and_theta(output_layer_weights_and_theta)
    
    # Iniciando os pesos e bias das camadas escondidas
    # hidden_layer_weights_and_theta -> Matriz contendo os valores predeterminados com os pesos e o bias das camadas escondidas, sendo que o bias é a ultima coluna da matriz
    def init_hidden_layer_weight_and_theta(self, hidden_layer_weights_and_theta):
        # Caso não há pesos e bias predeterminados pelo usuário 
        if not hidden_layer_weights_and_theta:
            
            #inicializa
            self.hidden_layer_weights_and_theta = []
            
            #inicializa primeira camada
            self.hidden_layer_weights_and_theta.append(np.random.uniform(-0.5, 0.5, (self.n_neurons_hiddens[0], self.n_neurons_input + 1))) 

            #para cada camada escondida que nao seja a primeira
            for i in range(1, self.n_hidden_layers):
                self.hidden_layer_weights_and_theta.append(np.random.uniform(-0.5, 0.5, (self.n_neurons_hiddens[i], self.n_neurons_hiddens[i-1]+1)))
         
        else:
            self.hidden_layer_weights_and_theta = hidden_layer_weights_and_theta

    # Iniciando os pesos e bias da camada de saida
    # output_layer_weights_and_theta -> Matriz contendo os valores predeterminados com os pesos e o bias da camada de saida, sendo que o bias é a ultima coluna da matriz
    def init_output_layer_weight_and_theta(self, output_layer_weights_and_theta):
        if not output_layer_weights_and_theta:
            self.output_layer_weights_and_theta = np.random.uniform(-0.5, 0.5, (self.n_neurons_output, self.n_neurons_hiddens[self.n_hidden_layers-1] + 1))
        else:
            self.output_layer_weights_and_theta = output_layer_weights_and_theta

    # Calculando os valores com a função de ativação, nesse caso a sigmoidal
    # net -> Sum(xi*wi) + theta(i)
    # theta = bias
    def activation_function(self, net):
        return (1/(1 + math.exp(-net)))
    
    # Calcula a derivada de f_net ou seja, da funcao de ativacao
    # f_net -> activation_function(Sum(xi*wi) + theta(i))
    def df_dnet(self, f_net):
        return (f_net * (np.array(1)-f_net))
    
    # Realiza o forward da rede neural
    def feed_forward(self, input_data):

        # Hidden Layer
        input_data.append(1)
        input_data = np.array(input_data)
    
        self.hidden_nets = []
        hidden_nets = []
        self.hidden_f_nets = []
        hidden_f_nets = []
        hidden_xi_wi = []

        #primeira camada escondida
        for i in range(self.hidden_layer_weights_and_theta[0].shape[0]):
            hidden_xi_wi.append(np.multiply(self.hidden_layer_weights_and_theta[0][i], input_data))
        
        for i in range(len(hidden_xi_wi)):
            hidden_nets.append(np.sum(hidden_xi_wi[i]))
            hidden_f_nets.append(self.activation_function(hidden_nets[i]))

        self.hidden_nets.append(hidden_nets)
        self.hidden_f_nets.append(hidden_f_nets)

        #demais camadas escondidas
        for j in range(1, self.n_hidden_layers):
            hidden_xi_wi=[]
            hidden_nets = []
            hidden_f_nets = []

            for i in range(self.hidden_layer_weights_and_theta[j].shape[0]):
                hidden_xi_wi.append(np.multiply(self.hidden_layer_weights_and_theta[j][i], np.append(self.hidden_f_nets[j-1],1)))

        
            for i in range(len(hidden_xi_wi)):
                hidden_nets.append(np.sum(hidden_xi_wi[i]))
                hidden_f_nets.append(self.activation_function(hidden_nets[i]))


            self.hidden_nets.append(hidden_nets)
            self.hidden_f_nets.append(hidden_f_nets)

        # Output Layer
        hidden_f_nets = hidden_f_nets.copy() #hidden f nets da ultima camada
        hidden_f_nets.append(1)
        self.output_nets = []
        self.output_f_nets = []
        output_xi_wi = np.multiply(self.output_layer_weights_and_theta, hidden_f_nets)
        for i in range(output_xi_wi.shape[0]):
            self.output_nets.append(np.sum(output_xi_wi[i]))
            self.output_f_nets.append(self.activation_function(self.output_nets[i]))

    
    """
    Função de treino utilizando o algoritmo backpropagation
    Parametros:
        dataset: dataset de entrada, como descrito em main.py
        eta: constante de treinamento
        threshold: limite de tolerancia do erro
        momentum: termo momentum
        max_iterations: numero maximo de iteracoes

    """
    def backpropagation(self, dataset, eta=0.3, threshold = 1e-3, momentum=0.5, max_iterations=30000):

        #numero de iteracoes
        it = 0
        aux_momentum = [0]*self.n_hidden_layers #momentum term for hidden layers
        output_momentum = 0 #momentum term for output

        squaredError = 2*threshold
        while(squaredError > threshold):
            squaredError = 0
            for i in range(len(dataset)):
                Xi =  dataset[i] [0:self.n_neurons_input]
                Yi =  dataset[i][self.n_neurons_input:len(dataset[0])]

                self.feed_forward(Xi)

                error = np.array(Yi) - np.array(self.output_f_nets)

                squaredError += np.sum(np.power(error, 2))

                output_delta = error * self.df_dnet(self.output_f_nets)

                #calculo dos hidden deltas

                #inicializa vetor de hidden delta
                hidden_delta = [0]*self.n_hidden_layers

                #delta para ultima camada escondida (usa pesos da camada de saida)
                hidden_delta[self.n_hidden_layers-1] = np.multiply(self.df_dnet(self.hidden_f_nets[self.n_hidden_layers-1]),
                    np.dot(np.matrix(output_delta), np.matrix(self.output_layer_weights_and_theta[:, 0:self.n_neurons_hiddens[self.n_hidden_layers-1]])))

            
                #para cada camada de neuronios, de tras para frente, com excecao da ultima
                for j in range(self.n_hidden_layers-2, -1, -1):
                    hidden_delta[j] = np.multiply(self.df_dnet(self.hidden_f_nets[j]),
                        np.dot(np.matrix(hidden_delta[j+1]), np.matrix(self.hidden_layer_weights_and_theta[j+1][:, 0:self.n_neurons_hiddens[j]])))
               

                #atualiza pesos da camada de saida
                output_aux = eta*(np.dot(np.transpose(np.matrix(output_delta)), np.matrix(np.append(self.hidden_f_nets[self.n_hidden_layers-1], 1))))
                self.output_layer_weights_and_theta = self.output_layer_weights_and_theta + output_aux
                self.output_layer_weights_and_theta = self.output_layer_weights_and_theta + output_momentum

                output_momentum = output_aux * momentum

                #atualizacao dos pesos das camadas escondidas
                #inicializa vetor auxiliar (de mudanca nos pesos)
                aux = [0]*self.n_hidden_layers

                #calculo para primeira camada, que utiliza input Xi
                aux[0] = eta*np.dot(np.matrix(hidden_delta[0]).T, np.matrix(Xi))
                self.hidden_layer_weights_and_theta[0] = np.add(self.hidden_layer_weights_and_theta[0], aux[0])
                #adds momentum term
                self.hidden_layer_weights_and_theta[0] = np.add(self.hidden_layer_weights_and_theta[0], aux_momentum[0])
                #updates momentum
                aux_momentum[0] = np.multiply(aux[0], momentum)
                
               
                #calculo para demais camadas esconidas
                for j in range(1, self.n_hidden_layers):
                    aux[j] = eta*np.dot(np.matrix(hidden_delta[j]).T, np.matrix(np.append(self.hidden_f_nets[j-1], 1)))
                    self.hidden_layer_weights_and_theta[j] = np.add(self.hidden_layer_weights_and_theta[j], aux[j])
                    #adds momentum term
                    self.hidden_layer_weights_and_theta[j] = np.add(self.hidden_layer_weights_and_theta[j], momentum*aux_momentum[j])
                    #updates momentum
                    aux_momentum[j] = np.multiply(aux[j], momentum)
                
                
            
            squaredError = squaredError/len(dataset)
            
            #imprime erro a cada 100 iteracoes
            if(it % 100 == 0):
                print("iteration",it," error: ",squaredError)
            it +=1

            if(it >= max_iterations):
                print("Maximum number of iterations (", max_iterations,") reached! - Final error:", squaredError)
                return

        print("Error threshold reached in iteration",it,"- Final error:", squaredError)



    # Mostra a rede neural
    def show(self):
        print("INPUT " + str(self.n_neurons_input))
        print("HIDDEN " + str(self.n_neurons_hiddens))
        print("OUTPUT " + str(self.n_neurons_output))

        print("--------HIDDEN LAYERS--------")
        for i in range (0, self.n_hidden_layers):
            print("--------LAYER ",i,"--------")
            print(self.hidden_layer_weights_and_theta[i])
            print()
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
    
    def show_class(self):
        print(self.output_f_nets)
