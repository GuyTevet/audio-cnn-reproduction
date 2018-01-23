

#!/usr/bin/env python

"""
    File name:          network_utils.py
    Author:             Guy Tevet & Chen Ponchek
    Date created:       23/1/2018
    Date last modified: 23/1/2018
    Description:        network utility fynctions
"""

import os
import numpy as np
import tensorflow as tf

NUM_CLASSES             = 46
INPUT_NUM_ROWS          = 256
INPUT_NUM_COLUMNS       = 96
INPUT_CHANNELS          = 1
CONV_KERNEL_SIZE        = 5
CONV1_OUTPUT_CHANNELS   = 64
CONV2_1_OUTPUT_CHANNELS = 64
CONV2_2_OUTPUT_CHANNELS = 128
NUM_HIDDEN_LAYERS       = 1024
BATCH_SIZE              = 32
DROPOUT_HOLD_PROB       = 0.9
LEARNING_RATE           = 0.0001
EPOCS                   = 1
K_FOLD                  = 10
TEMP_TESTSET_NUM = 4

def get_train_test_path(data_path, test_set_section_num):
    # GET TESET PATH
    testset_path = os.path.join(data_path, "classification_dataset01_0%0d.data" % test_set_section_num)

    # GET TRAINSET PATHES
    train_path = []
    for i in range(K_FOLD):
        if (i != test_set_section_num):
            train_path.append(os.path.join(data_path, "classification_dataset01_0%0d.data" % i))

    # DIVIDE PATHES TO SECTORS
    train_sectors = []
    train_sectors.append([train_path[0], train_path[1], train_path[2]])
    train_sectors.append([train_path[3], train_path[4], train_path[5]])
    train_sectors.append([train_path[6], train_path[7], train_path[8]])

    return [testset_path, train_sectors]

def get_testset(path):
    with open(path, 'rb') as file:
        testset = np.load(file, encoding='bytes')

    X = np.transpose(testset[0],(2,0,1))
    y = np.transpose(testset[1],(1,0))
    return X , y

def get_trainset_sector(sector_pathes):
    _trainset = []
    for path in sector_pathes:
        with open(path, 'rb') as file:
            _trainset.append(np.load(file, encoding='bytes'))

    X, y = _trainset[0][0], _trainset[0][1]
    for i in range(1,len(_trainset)):
        X = np.concatenate((X,_trainset[i][0]) , axis=2)
        y = np.concatenate((y, _trainset[i][1]) , axis=1)
    X = np.transpose(X,(2,0,1))
    y = np.transpose(y,(1,0))

    return X , y

def get_random_batch(batch_size , X_train , y_train) :
    rand_indices = np.random.randint(low=0,high=X_train.shape[0],size=batch_size)
    return np.expand_dims(X_train[rand_indices],axis=3), y_train[rand_indices]


# In[14]:


def get_random_test_batch(batch_size , X_test , y_test) :
    rand_indices = np.random.randint(low=0,high=X_test.shape[0],size=batch_size)
    return np.expand_dims(X_test[rand_indices],axis=3), y_test[rand_indices]

def get_incremental_batch(batch_size , batch_index , X , y) :
    return np.expand_dims(X[batch_size*(batch_index) : batch_size*(batch_index) + batch_size],axis=3), y[batch_size*(batch_index) : batch_size*(batch_index) + batch_size]

def get_almost_identical_batch(batch_size , X_train , y_train , y_batch_origin):

    switch_label_probability = 0.1

    y_train_labels = np.nonzero(y_train)[1]
    y_batch_origin_labels = np.nonzero(y_batch_origin)[1]

    for i in range(y_batch_origin_labels.shape[0]):
        rand = np.random.uniform() # [0,1)
        if rand < switch_label_probability :
            y_batch_origin_labels[i] = np.random.randint(46)

    y_batch = np.zeros([0 , 46] , dtype=np.float32)
    X_batch = np.zeros([0 , 256 , 96] , dtype=np.float32)
    for i in range(y_batch_origin.shape[0]):
        y_idx = np.where(y_train_labels == y_batch_origin_labels[i])
        random_idx = np.random.randint(low=0,high=y_idx[0].shape[0],size=1)
        X_batch = np.concatenate([X_batch , np.expand_dims(X_train[ y_idx[0][ random_idx[0] ] ],axis=0)])
        y_batch = np.concatenate([y_batch , np.expand_dims(y_train[ y_idx[0][ random_idx[0] ] ],axis=0)])

    return np.expand_dims(X_batch,axis=3), y_batch


def get_data_accuracy_loss(batch_size , X_test , y_test , sess , network , size_limit = -1) :

    if size_limit == -1 : #no limit
        iterations = X_test.shape[0] / batch_size
    else:
        iterations = min(X_test.shape[0],size_limit) / batch_size

    accuracy = 0
    loss = 0

    for i in range(iterations):

        matches = tf.equal(tf.argmax(network.y_pred, 1), tf.argmax(network.y_true, 1))
        acc_tensor = tf.reduce_mean(tf.cast(matches, tf.float32))

        #X , y = get_incremental_batch(batch_size,i,X_test,y_test)
        X , y = get_random_batch(batch_size , X_test , y_test)
        accuracy += sess.run(acc_tensor, feed_dict={network.x: X, network.y_true: y, network.keep_prob: 1.0,
                                 network.is_training: 'FALSE'})
        loss += network.cross_entropy_loss().eval(feed_dict={network.x:X, network.y_true:y, network.keep_prob:1.0, network.is_training:'FALSE'})

    accuracy = accuracy * 1.0 / iterations
    loss = loss * 1.0 / iterations

    return [accuracy , loss]


