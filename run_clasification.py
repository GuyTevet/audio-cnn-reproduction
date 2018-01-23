
#!/usr/bin/env python

"""
    File name:          run_clasification.py
    Author:             Guy Tevet & Chen Ponchek
    Date created:       23/1/2018
    Date last modified: 23/1/2018
    Description:        classification RUN script
                        change configurarion before running
"""


import tensorflow as tf
import numpy as np


# In[2]:


from clasification_cnn_network import *
from network_utils import *

# In[3]:


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
TEMP_TESTSET_NUM        = 4

#AGGREGATION_METHOD      = 'global_aggregation'
#AGGREGATION_METHOD      = 'avg_pooling'
#AGGREGATION_METHOD      = 'max_pooling'

#NETWORK_ARCH            = 'vanilla_cnn'
#NETWORK_ARCH            = 'resNeXt'
#DEEP_ARCH               = True

LOGDIR                  = './tensorboard'
LOG_NAME                = 'classification_vanilla_deep_maxpool_8000'
steps = 8001
arch = network_arch(type='vanilla_cnn' , deep=True , aggregation_method = 'max_pooling')
network = base_network(arch)

# In[5]:


from dataset_consts import *
from dataset_song_preprocess import *


####### SCRIPT START

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.80

data_path = "./datasets/preprocessed/classification_dataset01"
consts = Dataset_consts()
[testset_path, train_sectors_path] = get_train_test_path(data_path , TEMP_TESTSET_NUM)

[X_test , y_test] = get_testset(testset_path)


optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = optimizer.minimize(network.cross_entropy_loss())


init = tf.global_variables_initializer()



switch_sector = 100 #steps
train_loss_arr = []
test_loss_arr = []
train_acc_arr = []
test_acc_arr = []

next_train_sector = 0

#Set scalars for tensorboard
matches = tf.equal(tf.argmax(network.y_pred,1),tf.argmax(network.y_true,1)) 
acc = tf.reduce_mean(tf.cast(matches,tf.float32))
tf.summary.scalar("acc", acc)

#collect all summaries for tensorboard
summ = tf.summary.merge_all()

with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter(os.path.join(LOGDIR,LOG_NAME),graph=sess.graph)

    sess.run(init)
    for i in range(steps):


        if i % switch_sector == 0 :
            [X_train , y_train] = get_trainset_sector(train_sectors_path[next_train_sector])
            next_train_sector = (next_train_sector + 1) % len(train_sectors_path)

        x_batch,y_batch = get_random_batch(BATCH_SIZE , X_train , y_train)

        sess.run(train,feed_dict={network.x:x_batch, network.y_true:y_batch, network.keep_prob:DROPOUT_HOLD_PROB, network.is_training:'TRUE'})

        if i%50 == 0:
            print ("###Step : %0d"  %i)

        if i%20 == 0:
            s = sess.run(summ,
                         feed_dict={network.x: x_batch, network.y_true: y_batch, network.keep_prob: 1.0,
                                    network.is_training: 'FALSE'})
            writer.add_summary(s, i)
            writer.flush()


        if i%200 == 0 :
            #evaluate network
            #[train_acc , train_loss] = get_data_accuracy_loss(32 , X_train , y_train , sess , network  , size_limit=500)
            #[test_acc, test_loss] = get_data_accuracy_loss(32, X_test, y_test, sess, network, size_limit=500)

            train_loss = 0
            test_loss = 0
            train_acc = 0
            test_acc = 0

            iterations = 10

            for i in range(iterations):
                x_test_batch, y_test_batch = get_random_test_batch(BATCH_SIZE , X_test , y_test)
                x_train_batch, y_train_batch = get_random_test_batch(BATCH_SIZE, X_train, y_train)
                train_loss += network.cross_entropy_loss().eval(feed_dict={network.x:x_train_batch,      network.y_true:y_train_batch,      network.keep_prob:1.0, network.is_training:'FALSE'})
                test_loss  += network.cross_entropy_loss().eval(feed_dict={network.x:x_test_batch, network.y_true:y_test_batch, network.keep_prob:1.0, network.is_training:'FALSE'})

                train_acc += sess.run(acc ,feed_dict={network.x:x_train_batch,      network.y_true:y_train_batch,      network.keep_prob:1.0, network.is_training:'FALSE'})
                test_acc  += sess.run(acc ,feed_dict={network.x:x_test_batch, network.y_true:y_test_batch, network.keep_prob:1.0, network.is_training:'FALSE'})
                

            train_loss = train_loss / iterations
            test_loss = test_loss / iterations
            train_acc = train_acc / iterations
            test_acc = test_acc / iterations

            train_loss_arr.append(train_loss)
            test_loss_arr.append(test_loss)
            print("Train Loss is: %0.5f" % train_loss)
            print("Test  Loss is: %0.5f" % test_loss)

            train_acc_arr.append(train_acc)
            test_acc_arr.append(test_acc)
            
            print("Train Accuracy is: %0.5f" % train_acc)
            print("Test  Accuracy is : %0.5f" % test_acc)

            print('\n')

writer.close()