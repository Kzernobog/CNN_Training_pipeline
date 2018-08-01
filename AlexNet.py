# TODO:pause training 

import xml.etree.ElementTree as ET
import datetime
import sys
import re
import os
import h5py
from model import *
import numpy as np
import DL_utils as dl
import tensorflow as tf
from tqdm import tqdm, trange
import tracemalloc


class AlexNet(Model):
    """implements the alexnet model for image classification"""

    def __init__(self):
        pass

    def create_placeholders(self, n_H, n_W, n_C, n_y):
        """Function to create placeholder for tensorflow session
        params: n_H = height of the image
                n_W = width of image
                n_C = number of channels
                n_y = number of output features
        returns: X,Y """
        X = tf.placeholder(tf.float32, shape = (None, n_H, n_W, n_C))
        Y = tf.placeholder(tf.int32, shape = (n_y))
        return X, Y

    def forward_propagation(self, X, parameters):
        """Forward propagation for AlexNet without normalization
        params: X - image tensor
                parameters - weight tensor
        return: unactivate final layer nodes"""
        # 1st conv
        A1 = dl.conv_layer(X,parameters['W01'],parameters['b01'], [1,4,4,1],
                           padding='VALID', name='1')
        # 1st pool
        P1 = dl.max_pool(A1, kernel=[1,3,3,1], strides=[1,2,2,1], padding =
                         'VALID', name='1')
        # normalization
        norm1 = tf.nn.local_response_normalization(P1, name='norm1')
        # 2nd conv
        A2 = dl.conv_layer(norm1,parameters['W02'],parameters['b02'], strides
                           = [1,1,1,1], padding = 'SAME', name='2')
        #2nd pool
        P2 = dl.max_pool(A2, kernel=[1,3,3,1], strides=[1,2,2,1],
                                 padding = 'VALID', name='2')
        # normalization
        norm2 = tf.nn.local_response_normalization(P2, name='norm2')
        # 3rd conv
        A3 = dl.conv_layer(norm2,parameters['W03'],
                           parameters['b03'],
                           strides=[1,1,1,1],
                           padding='SAME', name='3')
        # 4th conv
        A4 = dl.conv_layer(A3,parameters['W04'],
                           parameters['b04'],
                           strides=[1,1,1,1],
                           padding = 'SAME',
                           name='4')
        # 5th conv
        A5 = dl.conv_layer(A4,parameters['W05'],
                          parameters['b05'],
                          strides=[1,1,1,1],
                          padding='SAME',
                          name='5')
        # 3rd pool
        P3 = dl.max_pool(A5,kernel=[1,3,3,1],strides=[1,2,2,1],padding ='VALID',name='3')
        # Flattening the last
        # pooling layer
        P3 = tf.contrib.layers.flatten(P3)
        # FC1 - 4096 neurons
        F1 = dl.fc_layer(P3,4096,activation_fn=None,name='1')
        # FC2 - 4096 neurons
        F2 = dl.fc_layer(F1,4096,activation_fn=None,name='2')
        # FC3 - 1000 neurons reping different classes - may be modified for other models
        F3 = dl.fc_layer(F2,1000,activation_fn=None, name='3')
        return F3

    def initialize_weights(self, xml_file):
        '''
        Reads model parameter weights from xml_file and initializes filters and biases
        params: xml_file - configuration xml with absolute path
        Return: parameters - a dictionary containing initialized parameters'''
        parameters = {}
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for child in root:
            size = []
            for child1 in child:
                # print(child.attrib['name'], child1.tag, child1.text)
                if (child1.tag == 'dimension'):
                    size.append((int)(child1.text))
                    size.append((int)(child1.text))
                if (child1.tag == 'input'):
                    size.append((int)(child1.text))
                if (child1.tag == 'output'):
                    size.append((int)(child1.text))
            W = tf.get_variable(child.attrib['name'], size, initializer = tf.glorot_uniform_initializer()) 
            parameters[child.attrib['name']] = W
            B = tf.get_variable('b'+(child.attrib['name'][1:]), [size[-1]], initializer = tf.zeros_initializer())
            parameters['b'+(child.attrib['name'][1:])] = B
            print(size, child.attrib['name'], 'b'+(child.attrib['name'][1:]))
        return parameters

    def compute_cost(self, Z, Y, batch_size, name='train'):
        ''' Computes cost
        params: Z8 -- Logits/Linear output from the last fully connected layer
                Y -- labels corresponding to each example in the batch
        Return: cost'''
        Y = tf.one_hot(Y, 1000)
        m = batch_size
        with tf.name_scope(name+"_loss"):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y))
            cost = cost/batch_size
            tf.summary.scalar(name+"_loss", cost, collections=[name])
            return cost

    def model(self, xml_path, LOGDIR, MODELDIR, test_path, train_path,
              learning_rate = 1e-5, num_epochs = 90, 
          minibatch_size = 128, print_cost = True):
        # restting the default graph
        tf.reset_default_graph()
        # global variables
        self.costs = []
        # creating placeholders 
        X, Y = self.create_placeholders(224,224,3,None)
        # initializing parameters
        parameters = self.initialize_weights(xml_path)
        # forward prop
        F3 = self.forward_propagation(X, parameters)
        # compute cost
        training_cost = self.compute_cost(F3, Y, minibatch_size)
        testing_cost = self.compute_cost(F3, Y, minibatch_size, name="test")
        # compute accuracy
        train_accuracy = dl.accuracy(F3, Y)
        test_accuracy = dl.accuracy(F3, Y, "test")
        # select the optimizer
        with tf.name_scope("train"):
            train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(training_cost)
        # initialize global variables
        init = tf.global_variables_initializer()
        # initialize the session
        with tf.Session() as sess:
            # run the initialization for the session
            sess.run(init)
            # initializing summaries, file writers and model saver
            train_merged_summary = tf.summary.merge_all('train')
            test_merged_summary = tf.summary.merge_all('test')
            train_writer = tf.summary.FileWriter(LOGDIR+"/train")
            test_writer = tf.summary.FileWriter(LOGDIR+"/test")
            train_writer.add_graph(sess.graph)
            test_writer.add_graph(sess.graph)
            saver = tf.train.Saver()
            # path to training folder
            PATH = train_path 
            # main training loop
            with h5py.File(PATH, mode='r') as h5_file:
                # number of examples
                (m, n_Htr, n_Wtr, n_Ctr) = h5_file['X_train'].shape
                test_epoch = 0
                # for loop for epoch/iterations
                for epoch in trange(num_epochs, desc="epochs"):
                    # maintain the cost through an epoch
                    epoch_cost = 0
                    try:
                        # ??? - REASON WHY
                        num_minibatches = int(m/minibatch_size)
                        for i in trange(num_minibatches, desc="minibatches"):
                            # procure minibatches
                            if (i == num_minibatches - 1):
                                minibatch_X = h5_file['X_train'][i*minibatch_size:]
                                minibatch_Y = h5_file['Y_train'][i*minibatch_size:]
                            else:
                                minibatch_X = h5_file['X_train'][i*minibatch_size:i*minibatch_size+minibatch_size]
                                minibatch_Y = h5_file['Y_train'][i*minibatch_size:i*minibatch_size+minibatch_size]
                            # optimize for cost
                            _ , minibatch_cost, _ , train_summary = sess.run([train_step, testing_cost, train_accuracy, train_merged_summary], feed_dict={X: minibatch_X, Y: minibatch_Y })
                            train_writer.add_summary(train_summary, epoch*num_minibatches + i)
                            # cumulative minibatch cost
                            epoch_cost += minibatch_cost/num_minibatches
                        # calculate the testing accuracy
                        if epoch % 5  == 0 and epoch != 0:

                            # testing accuracy
                            with h5py.File(test_path, mode='r') as h5_file_test:
                                number_of_test_images = h5_file_test['Y_val'].shape[0]
                                num_of_test_batches = int(number_of_test_images/minibatch_size)
                                tot_testing = 0
                                for i in trange(num_of_test_batches, desc="testing_batches"):
                                    if (i == num_of_test_batches):
                                        X_test = h5_file['X_train'][i*minibatch_size:]
                                        Y_test = h5_file['Y_train'][i*minibatch_size:]
                                    else:
                                        X_test = h5_file_test['X_val'][i*minibatch_size:i*minibatch_size+minibatch_size,:,:,:]
                                        Y_test = h5_file_test['Y_val'][i*minibatch_size:i*minibatch_size+minibatch_size]
                                    _ , test_loss, test_summary = sess.run([test_accuracy, testing_cost, test_merged_summary], feed_dict={X: X_test, Y: Y_test})
                                    test_writer.add_summary(test_summary, test_epoch*num_of_test_batches + i)
                            save_path = saver.save(sess, MODELDIR)
                            test_epoch += 1
                    except KeyboardInterrupt:
                        print("Saving model......")
                        save_path = saver.save(sess, MODELDIR)
                        sys.exit()
            return parameters


if __name__ == "__main__":
    day = datetime.date.today()
    test_path = "/home/aditya/Documents/Projects/atgm_vision_module/CNN-Implementations/data/h5_FILES_ImageNet/validation_folder/val.hdf5"
    LOGDIR = f'/home/aditya/Documents/Projects/indie_R/training_logs/alexnet/{day}'
    model_parent = f'/home/aditya/Documents/Projects/indie_R/weights/alexnet/{day}/'
    if not os.path.exists(model_parent):
        os.makedirs(model_parent)
    MODELDIR = model_parent+'model.ckpt'
    xml_path = 'AlexNet.xml'
    train_path = '/home/aditya/Documents/Projects/atgm_vision_module/CNN-Implementations/data/h5_FILES_ImageNet/training_folder/train.hdf5'
    model = AlexNet()
    model.model(xml_path, LOGDIR, MODELDIR, test_path, train_path)
