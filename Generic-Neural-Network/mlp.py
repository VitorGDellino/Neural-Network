import numpy as np

import math


class Mlp:
    # Criando a Rede MLP
    # n_neurons_input -> Numero de neuronios na camada de entrada, ou seja o tamanho da entrada
    # n_neurons_hiddens -> Numero de neuronios nas camadas escondidas
    # n_neurons_output -> Numero de neuronios na camada de saida, ou seja o tamanho da saida
    # Os parâmetros abaixo nao são necessários, pois eles sao gerados aletaoriamente usando valores entre -0.5 e 0.5 (Valores positivos e negativos ajudam na convergencia)
    # hidden_layer_weights_and_theta -> Matriz contendo os valores predeterminados com os pesos e o bias das camadas escondidas, sendo que o bias é a ultima coluna da matriz
    # hidden_layer_weights_and_theta -> Matriz contendo os valores predeterminados com os pesos e o bias da camada de saida, sendo que o bias é a ultima coluna da matriz
    def __init__(self, n_neurons_input, n_neurons_hiddens, n_neurons_output, hidden_layer_weights_and_theta = None, output_layer_weights_and_theta = None):
        # Setando o numero de neuronios em cada camada
        self.n_neurons_input = math.ceil(n_neurons_input)
        self.n_neurons_hiddens = math.ceil(n_neurons_hiddens)
        self.n_neurons_output = math.ceil(n_neurons_output)

        # Setando os pesos e o bias nas camadas escondida e de saida, respectivamente
        self.init_hidden_layer_weight_and_theta(hidden_layer_weights_and_theta)
        self.init_output_layer_weight_and_theta(output_layer_weights_and_theta)
    
    # Iniciando os pesos e bias das camadas escondidas
    # hidden_layer_weights_and_theta -> Matriz contendo os valores predeterminados com os pesos e o bias das camadas escondidas, sendo que o bias é a ultima coluna da matriz
    def init_hidden_layer_weight_and_theta(self, hidden_layer_weights_and_theta):
        # Caso não há pesos e bias predeterminados pelo usuário 
        if not hidden_layer_weights_and_theta:
            self.hidden_layer_weights_and_theta = np.random.uniform(-0.5, 0.5, (self.n_neurons_hiddens, self.n_neurons_input + 1))
        else:
            self.hidden_layer_weights_and_theta = hidden_layer_weights_and_theta

    # Iniciando os pesos e bias da camada de saida
    # output_layer_weights_and_theta -> Matriz contendo os valores predeterminados com os pesos e o bias da camada de saida, sendo que o bias é a ultima coluna da matriz
    def init_output_layer_weight_and_theta(self, output_layer_weights_and_theta):
        if not output_layer_weights_and_theta:
            self.output_layer_weights_and_theta = np.random.uniform(-0.5, 0.5, (self.n_neurons_output, self.n_neurons_hiddens + 1))
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
        return (f_net * (1-f_net))
    
    # Realiza o forward da rede neural
    def feed_forward(self, input_data):

        # Hidden Layer
        input_data.append(1)
        self.hidden_nets = []
        self.hidden_f_net = hidden_f_net = []
        hidden_xi_wi = np.multiply(self.hidden_layer_weights_and_theta, input_data)
        for i in range(hidden_xi_wi.shape[0]):
            self.hidden_nets.append(np.sum(hidden_xi_wi[i]))
            hidden_f_net.append(self.activation_function(self.hidden_nets[i]))

        self.hidden_f_net = np.copy(hidden_f_net)

        # Output Layer
        hidden_f_net.append(1)
        self.output_nets = []
        self.output_f_nets = []
        output_xi_wi = np.multiply(self.output_layer_weights_and_theta, hidden_f_net)
        for i in range(output_xi_wi.shape[0]):
            self.output_nets.append(np.sum(output_xi_wi[i]))
            self.output_f_nets.append(self.activation_function(self.output_nets[i]))
    
    # Mostra a rede neural
    def show(self):
        print("INPUT " + str(self.n_neurons_input))
        print("HIDDEN " + str(self.n_neurons_hiddens))
        print("OUTPUT " + str(self.n_neurons_output))

        print("--------HIDDEN LAYER--------")
        print(self.hidden_layer_weights_and_theta)
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
        print(self.hidden_f_net)
        print()
        print("OUTPUT NET")
        print(self.output_nets)
        print()
        print("OUTPUT F_NET")
        print(self.output_f_nets)
