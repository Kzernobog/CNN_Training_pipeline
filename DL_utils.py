import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
import math


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def conv_layer(A_p, W, B, strides=[1,1,1,1], padding="SAME", name="default", activation='relu'):
    '''
    A_p = activation of the previous layer
    W = Filter to convolve
    B = bias term
    acitvation = type of activation
    '''
    #tf.name_scope creates namespace for operators in the default graph, places into group, easier to read
    with tf.name_scope('train_conv_'+name):
        conv = tf.nn.conv2d(A_p, W, strides=strides, padding=padding)

        if (activation == 'relu'):
            act = tf.nn.relu(tf.nn.bias_add(conv, B))
        elif (activation == 'leaky_relu'):
            act = tf.nn.leaky_relu(tf.nn.bias_add(conv, B))


        #visualize the the distribution of weights, biases and activations
        tf.summary.histogram("weights", W, collections=['train'])
        tf.summary.histogram("biases", B, collections=['train'])
        tf.summary.histogram("activations", act, collections=['train'])
        #return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        return act

def fc_layer(A_p, output_num, activation_fn=None, name="default"):
    """
    A_p = activations of the previous layer
    output_num = number of neurons in the fully connected layer
    """
    with tf.name_scope("train_fc_"+name):
        #fully connected part
        FC1 = tf.contrib.layers.fully_connected(A_p, output_num, activation_fn=activation_fn)
        return FC1

def max_pool(A_p, kernel, strides, padding="SAME", name="default"):
    """
    A_p = activation of the previous layer
    kernel = size of the filter
    strides = strides of the pooling filter
    """
    with tf.name_scope("train_max_pool_"+name):
        P = tf.nn.max_pool(A_p, kernel, strides, padding=padding)
        return P
    
def avg_pool(A_p, kernel, strides, padding='SAME', name='default'):
    '''
    A_p = activation of the previous layer
    kernel = size of the filter
    strides = strides of the pooling filter
    '''
    with tf.name_scope('train_avg_pool_'+name):
        P = tf.nn.avg_pool(A_p, kernel, strides, padding=padding)
        return P

def dropout(X, keep_prob, name='default'):
    '''
    X = input tensor
    keep_prob = probability that each element is kept
    '''
    with tf.name_scope('train_dropout_'+name):
        res = tf.nn.dropout(X, keep_prob)
        return res


# util function for GoogLeNet/InceptionNet
def inception_module(A_p, parameters, name='default'):
    '''
    Calculates the ouput for an inception module - util function for the GoogLeNet model
    A_P = activations of the previous layer
    paramters = parameters/weights
    '''
    with tf.name_scope("Inception_Module_"+name):
        A1 = conv_layer(A_p, parameters['W1'], name='1')
        A2 = conv_layer(A_p, parameters['W2'], name='2')
        P1 = max_pool(A_p, kernel=[1,3,3,1], name='1')
        
        A_1x1 = conv_layer(A_p, parameters['W_1x1'], name='_1x1')
        A_3x3 = conv_layer(A1, parameters['W_3x3'], name='_3x3')
        A_5x5 = conv_layer(A2, parameters['W_5x5'], name='_5x5')
        A_p1x1 = conv_layer(P1, parameters['W_p1x1'], name='_p1x1')
        
        return rf.concat([A_1x1, A_3x3, A_5x5, A_p1x1], axis=3)
    
def accuracy(y_hat, y, data_use='train'):
    '''
    calculates the accuracy
    '''
    y = tf.one_hot(y, 1000)
    with tf.name_scope("accuracy_"+data_use):
        predict_op = tf.argmax(y_hat, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        if (data_use=='train'):
            tf.summary.scalar(data_use+"_accuracy", accuracy, collections=['train'])
        else:
            tf.summary.scalar(data_use+"_accuracy", accuracy, collections=['test'])
        return accuracy

def main():
    return None

if __name__ == "__main__":
    main()
