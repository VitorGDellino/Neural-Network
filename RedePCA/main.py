from pca_network import PCA_Network
import sklearn.datasets as dt
import numpy as np
import tensorflow as tf
#from tensorflow import keras

N_COMPONENTS = 3

def main():

	pca_n = PCA_Network(13, N_COMPONENTS)

	data, target = dt.load_wine(return_X_y=True)
	data = np.matrix(data)

	norm = (data - np.mean(data, axis=0))/np.std(data, axis=0)

	
	#pca_n.train(norm)

	print("PCA adaptativa:")
	print(np.transpose(pca_n.input_weights))
	print()

	_, _, eig_vecs = pca(norm)
	print("PCA")
	print(eig_vecs[:,:N_COMPONENTS])
	print()

	#projetar dados via PCA adaptativa
	output = feed_net(norm, pca_n)

	#treinar e testar via MPL
	#	dados originais (normalizados)
	#	dados projetados


def feed_net(data, pca_n):

	output = np.zeros((len(data), N_COMPONENTS))
	#print(output)

	for i in range(0, len(data)):
		output[i] = np.transpose(pca_n.feed_forward(data[i]))

	return output



def pca(data):
	"""
	Principal components analysis

	Parameters:
		data - numpy array with the data
	
	Return:
		proj_data - data in the new coordinates space
		eig_vals - eigen values, ordered in descending order
		eig_vecs - eigen vectors associated with the eig_vals
	"""
	#standardization
	norm = (data - np.mean(data, axis=0))/np.std(data, axis=0)

	#covariance matrix 
	cov_m = np.dot(norm.transpose(),norm)/(len(norm)-1)

	eig_vals, eig_vecs = np.linalg.eig(cov_m)

	#order eig values and eig vecs in descending order
	order = np.argsort(-eig_vals)
	eig_vals = eig_vals[order]
	eig_vecs = eig_vecs[:,order]

	#project data into new coordinates system
	proj_data = np.dot(norm,eig_vecs)

	return proj_data, eig_vals, eig_vecs



if __name__ == '__main__':
	main()