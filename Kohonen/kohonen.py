import numpy as np
import math
import sklearn.datasets as dt
from matplotlib import pyplot as plt
from scipy.misc import toimage
from PIL import Image

def eucledian_distance(x, y):
        return np.sqrt(np.sum(np.power((x - y),2)))

def update_learning_rate(lr, iter, max_iter):
        return lr * math.exp((-iter)/max_iter)

def bmu(x, y):
        dist_matrix = np.zeros((x.shape[0], x.shape[1]))
        for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                        dist_matrix[i, j] = eucledian_distance(x[i, j], y)

        index = np.argmin(dist_matrix)
        index =  int(index/x.shape[0]), int(index % x.shape[1])
        #print(np.mean(dist_matrix))
        return index

def update_neighbours(index, weights, neighbourhood, input_data, alpha, sigma):
        neighbourhood = neighbourhood + 1
        dir = [[1,0], [1,1], [0,1], [-1, 1], [-1,0], [-1,-1], [0,-1], [1,-1]]

        #update weights in index
        weights[index] = weights[index] + alpha*(input_data - weights[index])

        #update weights in neghbourhood
        for i in range(neighbourhood):
                for j in range(len(dir)):
                        x = (i+1)*dir[j][0] + index[0]
                        y = (i+1)*dir[j][1] + index[1]
                        if(x < weights.shape[0] and x >= 0):
                                if(y < weights.shape[1] and y >= 0):
                                        h = math.exp(-(eucledian_distance(np.array([x, y]), index))/(2* (sigma**2)))
                                        weights[x][y] = weights[x][y] +  h*alpha*(input_data - weights[x, y])

        
        
def init_weight(nattributes, nrow=10, ncol=10):
        return np.random.rand(nrow, ncol, nattributes)

def train(data, weights, max_iter=100, alpha=0.5, sigma=0.5):
        iter = 0

        while(iter < max_iter):
                for row in data:
                        index = bmu(weights, row)
                        update_neighbours(index, weights, 2, row, alpha, sigma)
                alpha = update_learning_rate(alpha, iter, max_iter)
                sigma = update_learning_rate(sigma, iter, max_iter)
                iter = iter + 1
                print("iteration " + str(iter) + "/" + str(max_iter) + "  ALPHA = " + str(alpha) + " SIGMA = " + str(sigma))

def test(weights, data, classes):
        result_map = np.zeros((weights.shape[0], weights.shape[1], 3))
        i = 0
        for row in data:
                index = bmu(weights, row)
                if classes[i] == 0:
                        if result_map[index][0] <= 0.9:
                                result_map[index] += np.asarray([0.1,0,0])
                elif classes[i] == 1:
                        if result_map[index][1] <= 0.9:
                                result_map[index] += np.asarray([0,0.1,0])
                elif classes[i] == 2:
                        if result_map[index][2] <= 0.9:
                                result_map[index] += np.asarray([0,0,0.1])
                i+=1

        result_map = np.flip(result_map,0)

        plt.imshow(result_map)
        plt.show()

def main():

    (data, target) = dt.load_wine(return_X_y=True)
    data = np.asmatrix(data)
    data = (data - np.amin(data))/(np.amax(data) - np.amin(data))
    weights = init_weight(data.shape[1], nrow=7, ncol=7)
    train(data, weights, max_iter = 100)
    weights = (weights - np.amin(weights))/(np.amax(weights) - np.amin(weights))
    test(weights, data, target)

    

if __name__ == "__main__":
    main()