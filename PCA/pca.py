import sklearn.datasets as dt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
from matplotlib import pyplot as plt


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


def main():

	#load iris dataset
	data, target = dt.load_iris(return_X_y=True)
	data = np.asarray(data)

	#get pca of data
	proj_data, eig_vals, eig_vecs = pca(data)

	#component contribution analysis
	eig_vals_norm = eig_vals/sum(eig_vals)

	plt.bar(range(1,5), eig_vals_norm, alpha=0.5, align='center', label='Contribuição individual')
	plt.step(range(1,5), np.cumsum(eig_vals_norm), where='mid', label='Contribuição acumulada')
	plt.legend(loc='best')
	plt.xlabel("Componentes")
	plt.ylabel("Razão de variância explicada")
	plt.title("Contribuição das componentes")
	plt.tight_layout()
	plt.show()

	#plot projection with the 2 first components
	fig, ax = plt.subplots()

	color = ['#1b9e77', '#d95f02', '#7570b3']
	for idx, val in enumerate(np.unique(target)):
	    index = np.argwhere(target==val)
	    ax.scatter(proj_data[index,0], proj_data[index,1], c = color[idx], label=val)


	ax.legend()
	plt.title("Projeção sobre as duas principais componentes")
	plt.tight_layout()
	plt.show()

	#Plot projection with the 3 first components
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	color = ['#1b9e77', '#d95f02', '#7570b3']
	for idx, val in enumerate(np.unique(target)):
	    index = np.argwhere(target==val)
	    ax.scatter(proj_data[index,0], proj_data[index,2], proj_data[index,1], c = color[idx], label=val)


	ax.legend()
	plt.title("Projeção sobre as três principais componentes")
	plt.show()


if __name__ == "__main__":
    main()
