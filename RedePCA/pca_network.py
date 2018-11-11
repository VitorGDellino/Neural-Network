import numpy as np

class PCA_Network:

	def __init__(self, n_neurons_input, n_neurons_output):

		self.n_neurons_input = n_neurons_input
		self.n_neurons_output = n_neurons_output

		self.init_weights()


	def init_weights(self):

		#inner weights
		self.input_weights = np.random.uniform(-0.5, 0.5, (self.n_neurons_output, self.n_neurons_input))
		self.input_weights = (self.input_weights - np.min(self.input_weights))/(np.max(self.input_weights) - np.min(self.input_weights))

		norm = np.linalg.norm(self.input_weights, axis=1)
		self.input_weights /= norm[:,None]

		#side weights
		self.neuron_weights = np.random.uniform(-0.5, 0.5, (self.n_neurons_output, self.n_neurons_output))
		self.neuron_weights = np.tril(self.neuron_weights, -1) #zeroes upper triangle + diagonal


	def feed_forward(self, input_data):
		"""
		Feed input into adaptive PCA Network

		Parameters:
			input_data: array containing input data
		"""

		#turn it into a matrix so it can be transposed
		#input_data = np.matrix(input_data)
		
		#inner weights
		self.output = np.dot(self.input_weights, np.transpose(input_data))

		for i in range(1, self.n_neurons_output):
			self.output[i] += np.dot(self.neuron_weights[i], self.output)

		return self.output





	def train(self, input_data, threshold = 0.0001, max_iterations = 2500, learning_rate=0.1, side_learning_rate=0.05, momentum=0.1):

		n_examples, _ = np.shape(input_data)
		momentum_input_weights = np.zeros(np.shape(self.input_weights))
		momentum_neuron_weights = np.zeros(np.shape(self.neuron_weights))

		#input_data = np.matrix(input_data)

		alpha = 0.99
		

		for c in range(0, max_iterations):

			for i in range(0, n_examples):

				self.feed_forward(input_data[i])
			

				#update inner weights
				#xiyi+momentum
				delta = learning_rate*np.dot(self.output, input_data[i])+momentum*momentum_input_weights

				self.input_weights += delta
				momentum_input_weights = delta

				#normalize per row
				norm = np.linalg.norm(self.input_weights, axis=1)
				self.input_weights /= norm[:,None]

				#update side weights
				#yl*yj
				aux = np.dot(self.output, np.transpose(self.output))
				aux = np.tril(aux, -1)
				delta = -side_learning_rate*aux + momentum*momentum_neuron_weights

				self.neuron_weights += delta
				momentum_neuron_weights = delta


				if(np.amax(np.abs(self.neuron_weights)) < threshold):
					break

				#update training parameters
				momentum = max(alpha*momentum, 0.0001)
				learning_rate = max(alpha*learning_rate, 0.0001)
				side_learning_rate = max(alpha*side_learning_rate, 0.0002)
				#print(np.amax(np.abs(self.neuron_weights)))
				#input()

			if(np.amax(np.abs(self.neuron_weights)) < threshold):
				print("Threshold reached!", np.amax(np.abs(self.neuron_weights)) )
				break

			if(c%50==0):
				print("iteration", c,"-",np.amax(np.abs(self.neuron_weights)) )
				









		




