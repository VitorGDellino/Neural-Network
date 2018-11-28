import numpy as np
import sklearn.datasets as dt
from matplotlib import pyplot as plt

ALPHA = 0.5
EPSILON = 1.5

def bmu(x, y):
        dist_matrix = np.zeros((x.shape[0], x.shape[1]))
        for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                        dist_matrix[i, j] = np.sqrt(np.sum(np.power((x[i, j] - y),2)))

        index = np.argmin(dist_matrix)
        index =  int(index/x.shape[0]), int(index % x.shape[1])
        
        return index

def update_neighbours(index, weights, neighbourhood, input_data):
        neighbourhood = neighbourhood + 1
        dir = [[1,0], [1,1], [0,1], [-1, 1], [-1,0], [-1,-1], [0,-1], [1,-1]]

        #update weights in index
        weights[index] = weights[index] + ALPHA*(input_data - weights[index])

        #update weights in neghbourhood
        for i in range(neighbourhood):
                for j in range(len(dir)):
                        x = (i+1)*dir[j][0] + index[0]
                        y = (i+1)*dir[j][1] + index[1]
                        if(x < weights.shape[0] and x >= 0):
                                if(y < weights.shape[1] and y >= 0):
                                        #print((i+1)*dir[j][0] + index[0], (i+1)*dir[j][1] + index[1])
                                        weights[x][y] = weights[x][y] + (1/(i+1) + EPSILON)*ALPHA*(input_data - weights[x, y])

        
        
def init_weight(nattributes, nrow=10, ncol=10):
        return np.random.rand(nrow, ncol, nattributes)

def train(data, weights, max_iter=100):
        iter = 0

        while(iter < max_iter):
                for row in data:
                        index = bmu(weights, row)
                        update_neighbours(index, weights, 1, row)

                iter = iter + 1
                print("iteration " + str(iter) + "/" + str(max_iter))

def test():
        None

def main():

    (data, target) = dt.load_wine(return_X_y=True)
    data = np.asmatrix(data)
    weights = init_weight(data.shape[1], nrow=3, ncol=3)
    print(weights)
    train(data, weights)
    weights = (weights - np.amin(weights))/(np.amax(weights) - np.amin(weights))
    print(weights)

    

if __name__ == "__main__":
    main()