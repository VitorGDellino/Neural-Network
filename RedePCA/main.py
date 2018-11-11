from pca_network import PCA_Network
import sklearn.datasets as dt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

N_COMPONENTS = 8

def main():

    #get wine dataset
    data, target = dt.load_wine(return_X_y=True)
    data = np.matrix(data)

    #normalize it
    norm = (data - np.mean(data, axis=0))/np.std(data, axis=0)

    #initialize and train PCA network
    pca_n = PCA_Network(13, N_COMPONENTS)
    pca_n.train(norm)
    
    #PCA from network
    print("PCA adaptativa:")
    print(np.transpose(pca_n.input_weights))
    print()

    #Traditional PCA
    _, _, eig_vecs = pca(norm)
    print("PCA")
    print(eig_vecs[:,:N_COMPONENTS])
    print()


    fig, ax = plt.subplots()
    proj_data = np.dot(norm,eig_vecs)
    color = ['#1b9e77', '#d95f02', '#7570b3']
    for idx, val in enumerate(np.unique(target)):
        index = np.argwhere(target==val)
        ax.scatter([proj_data[index,0]], [proj_data[index,1]], c = color[idx], label=val)


    ax.legend()
    plt.title("PCA - Projeção sobre as duas principais componentes")
    plt.tight_layout()
    plt.show()


    #project data with adaptive PCA
    output = feed_net(norm, pca_n)

    fig, ax = plt.subplots()
    color = ['#1b9e77', '#d95f02', '#7570b3']
    for idx, val in enumerate(np.unique(target)):
        index = np.argwhere(target==val)
        ax.scatter([output[index,0]], [output[index,1]], c = color[idx], label=val)


    ax.legend()
    plt.title("PCA Adaptativa - Projeção sobre as duas principais componentes")
    plt.tight_layout()
    plt.show()


    #formating output for use in network
    target_cat =  keras.utils.to_categorical(target, 3)


    #Train and test with MLPs
    #	the original data
    train_test_original_dataset(norm, target_cat)

    #	projected data
    train_test_pca(output, target_cat)



def feed_net(data, pca_n):
    """
    Feed all the rows of data to the pca network and get
    the projected data output

    """
    output = np.zeros((len(data), N_COMPONENTS))

    for i in range(0, len(data)):
        output[i] = np.transpose(pca_n.feed_forward(data[i]))

    return output


def train_test_original_dataset(norm, target_cat):
    """
    Train a MLP network on the original data and get accuracy
    """
    x_train, x_test, y_train, y_test = train_test_split(norm, target_cat, test_size=0.15)

    model = keras.Sequential()
    model.add(keras.layers.Dense(10, input_dim=13, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(6, activation='relu'))
    model.add(keras.layers.Dense(6, activation='relu'))
    model.add(keras.layers.Dense(4, activation='relu'))
    model.add(keras.layers.Dense(2, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))


    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=15, epochs=1000, verbose=0)

    test_loss, test_acc = model.evaluate(x_test, y_test)

    print('Original dataset - test accuracy:', test_acc)


def train_test_pca(pca_data, target_cat):
    """
    Train a MLP network on pca-projected data and get accuracy
    """
    x_train, x_test, y_train, y_test = train_test_split(pca_data, target_cat, test_size=0.15)

    model = keras.Sequential()
    model.add(keras.layers.Dense(10, input_dim=N_COMPONENTS, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(6, activation='relu'))
    model.add(keras.layers.Dense(6, activation='relu'))
    model.add(keras.layers.Dense(4, activation='relu'))
    model.add(keras.layers.Dense(2, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))

    

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=15, epochs=1000, verbose=0)

    test_loss, test_acc = model.evaluate(x_test, y_test)

    print('PCA dataset - test accuracy:', test_acc)

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