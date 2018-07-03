import h5py
import matplotlib.pyplot as plt
import scipy
import math
import numpy as np
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops

#from rnn_utils import *


def create_placeholders(n_x,n_y,n_a,T):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, number of input features
    n_y -- scalar, number of classes
    T   -- scalar, number of periods
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None, 6] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_x, None, 6] and dtype "float"
    """
    
    X = tf.placeholder(tf.float32, [n_x, None, T], name = "X_inputs")
    Y = tf.placeholder(tf.float32, [n_y, None, T], name = "Y_slate")
    A0 = tf.placeholder(tf.float32, [n_a, None], name = "A_HF")
    
    
    return X, Y, A0


def initialize_parameters(n_x,n_a,n_y):
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        Wax : [n_a, n_x]
                        Waa : [n_a, n_a]
                        Wya : [n_y, n_a]
                        ba : [n_a, 1]
                        by : [n_y, 1]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    
    tf.set_random_seed(1)                              # so that your "random" numbers match ours
    
    #RNN Parameters
    Wax = tf.get_variable("Wax", [n_a, n_x], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    Waa = tf.get_variable("Waa", [n_a, n_a], initializer = tf.zeros_initializer())
    Wya = tf.get_variable("Wya", [n_y, n_a], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    ba = tf.get_variable("ba", [n_a, 1], initializer = tf.zeros_initializer())
    by = tf.get_variable("by", [n_y, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    #LSTM Parameters
    
    parameters = {"Wax": Wax,
                  "Waa": Waa,
                  "Wya": Wya,
                  "ba":  ba,
                  "by":  by}
    
    return parameters


def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell as described in Figure (2)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """
    
    # Retrieve parameters from "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    ### START CODE HERE ### (≈2 lines)
    # compute next activation state using the formula given above
    a_next = tf.sigmoid(tf.add(tf.add(tf.matmul(Wax, xt), tf.matmul(Waa, a_prev)), ba))
    # compute output of the current cell using the formula given above
    yt_pred = tf.add(tf.matmul(Wya, a_next), by)
    yt_pred = tf.nn.softmax(yt_pred, 0)

    ### END CODE HERE ###
    
    # store values you need for backward propagation in cache
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, yt_pred, cache



def rnn_forward(x, A0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """
    
    # Initialize "caches" which will contain the list of all caches
    caches = []
    
    # Retrieve dimensions from shapes of x and Wya
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    
        # Initialize a_next (≈1 line)
    a_next = A0
    a_list = []
    y_list = []
    
    # loop over all time-steps
    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        a_list.append(a_next)
        y_list.append(yt_pred)
        
        # Append "cache" to "caches" (≈1 line)
        caches.append(cache)

    a = tf.stack(a_list)
    y_pred = tf.stack(y_list)
    
    a = tf.transpose(a, [1, 2, 0])
    y_pred = tf.transpose(y_pred, [1, 2, 0])
    
    ### END CODE HERE ###
    
    # store values needed for backward propagation in cache
    caches = (caches, x)
    
    return a, y_pred, caches


def compute_cost(y_pred, Y):
    """
    Computes the cost
    
    Arguments:
    y_pred -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
#    logits = tf.transpose(y_pred)
#    labels = tf.transpose(Y)
    logits = y_pred
    labels = Y
    
    print(y_pred.shape)
    print(Y.shape)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    return cost


def random_mini_batches(X, Y, mini_batch_size, seed):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector, of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    n_y = Y.shape[0] 
    T = X.shape[2]
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation,:]
    shuffled_Y = Y[:, permutation,:].reshape((n_y,m,T))
    
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size,:]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size,:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:

#        end = 10000 - mini_batch_size * math.floor(10000 / mini_batch_size)
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:,:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:,:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches




def neural_net_rnn(X_train, Y_train, X_test, Y_test, n_a, learning_rate , num_epochs, minibatch_size, print_cost = True):
    
    """
    Implements a tensorflow recurrent neural network with 8 hidden features:
    
    Arguments:
    X_train -- Training X set
    Y_train -- Train Y set
    X_test -- Test X set
    Y_test -- Test Y set
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch, mini batch sizes should be powers of 2.
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """


    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    n_x = X_train.shape[0]
    n_y = Y_train.shape[0]
    m = X_train.shape[1]
    T = X_train.shape[2]
    n_a = n_a
    costs = []                                        # To keep track of the cost
    
    
    
    
    # Create Placeholders of shape (n_x, n_y)
    
    X, Y, A0 = create_placeholders(n_x, n_y, n_a, T)
    
    # Initialize parameters
    parameters = initialize_parameters(n_x,n_a,n_y)
    
    a, y_pred, caches= rnn_forward(X, A0, parameters)
    
    cost = compute_cost(y_pred, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()
        
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):
            
            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            for minibatch in minibatches:
                
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, A0: np.random.randn(n_a,minibatch_X.shape[1])})
                
                epoch_cost += minibatch_cost / num_minibatches
            
            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per 5)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        
        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(y_pred), tf.argmax(Y))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train, A0: np.random.randn(n_a,X_train.shape[1])}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test, A0: np.random.randn(n_a,X_test.shape[1])}))
        
        train_r = accuracy.eval({X: X_train, Y: Y_train, A0: np.random.randn(n_a,X_train.shape[1])})
        test_r  = accuracy.eval({X: X_test, Y: Y_test, A0: np.random.randn(n_a,X_test.shape[1])})
            
        return parameters, epoch_cost, train_r, test_r
    
