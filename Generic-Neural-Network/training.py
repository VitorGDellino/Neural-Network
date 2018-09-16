import pandas as pd
import numpy as np

def holdout(p, dataframe):
    """
    Separate dataframe randomly into training
    and testing sets

    Parameters:
        p: percentage of datagrame used for training
        dataframe: pandas dataframe that should be  divided

    Return:
        testing_set
        training_set
    """


    train=dataframe.sample(frac=p)
    test=dataframe.drop(train.index)

    return (train, test)


def accuracy(neural_network, test_data, n_classes):
    """
    Feed examples to neural network and get accuracy of classification.

    Class must be in the last columns of data.

    """
    n_correct = 0
    #feed each training example to neural network
    for example in test_data.values.tolist():
        
        #feed example
        neural_network.feed_forward(example[:(-1*n_classes)])

        #gets output from neural network and rounds it to int
        output = np.round(neural_network.output_f_nets)

        #compares expected result and output
        if(np.array_equal(example[(-1*n_classes):], output)):
            n_correct += 1

    #accuracy
    return (n_correct/len(test_data))


def squared_error(neural_network, test_data, n_classes):


    squared_error = 0
    #feed each training example to neural network
    for example in test_data.values.tolist():
        
        #feed example
        neural_network.feed_forward(example[:(-1*n_classes)])

        #gets output from neural network and rounds it to int
        output = neural_network.output_f_nets

        #compares expected result and output
        error = np.array(example[(-1*n_classes):]) - np.array(output)
        
        squared_error += np.sum(np.power(error, 2))

    #accuracy
    return (squared_error/len(test_data))

    