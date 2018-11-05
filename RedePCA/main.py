from pca_network import PCA_Network
import sklearn.datasets as dt
import numpy as np

def main():

	pca_n = PCA_Network(13, 3)

	data, target = dt.load_wine(return_X_y=True)
	data = np.matrix(data)

	norm = (data - np.mean(data, axis=0))/np.std(data, axis=0)

	#pca.feed_forward(data[0,:])
	pca_n.train(norm)

	print(np.transpose(pca_n.input_weights))

	_, _, eig_vecs = pca(norm)
	print(eig_vecs)


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