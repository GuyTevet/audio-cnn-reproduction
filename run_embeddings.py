
#!/usr/bin/env python

"""
    File name:          run_embeddings.py
    Author:             Guy Tevet & Chen Ponchek
    Date created:       23/1/2018
    Date last modified: 23/1/2018
    Description:        embeddings RUN script
                        change configurarion before running
"""


import tensorflow as tf
import numpy as np


# In[2]:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embeddings_siamese_network import siamese_network
from clasification_cnn_network import *
from network_utils import *
from dataset_consts import *
from dataset_song_preprocess import *
from sklearn.manifold import TSNE


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
TEMP_TESTSET_NUM = 4
EMBEDINGS_SIZE = 16
# In[4]:

#AGGREGATION_METHOD      = 'global_aggregation'
#AGGREGATION_METHOD      = 'avg_pooling'
#AGGREGATION_METHOD      = 'max_pooling'

#NETWORK_ARCH            = 'vanilla_cnn'
#NETWORK_ARCH            = 'resNeXt'
#DEEP_ARCH               = True

LOGDIR                  = './tensorboard'
LOG_NAME                = 'embeddings_vanilla_deep_maxpool_8000'
steps = 8001
arch = network_arch(type='vanilla_cnn' , deep=True , aggregation_method = 'max_pooling')
network = siamese_network(arch)


# In[6]:

data_path = "./datasets/preprocessed/classification_dataset01"
consts = Dataset_consts()
[testset_path, train_sectors_path] = get_train_test_path(data_path , TEMP_TESTSET_NUM)

[X_test , y_test] = get_testset(testset_path)

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = optimizer.minimize(network.loss)


# In[ ]:


init = tf.global_variables_initializer()


# In[ ]:


steps = 5001
train_loss_arr = []

switch_sector = 100 #steps
next_train_sector = 0

#collect all summaries for tensorboard
summ = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.join(LOGDIR,LOG_NAME),graph=sess.graph)
    
    sess.run(init)
    
    for i in range(steps):

        if i % switch_sector == 0 :
            [X_train , y_train] = get_trainset_sector(train_sectors_path[next_train_sector])
            next_train_sector = (next_train_sector + 1) % len(train_sectors_path)
        
        x_batch_1,y_batch_1 = get_random_batch(BATCH_SIZE , X_train , y_train)
        #x_batch_2,y_batch_2 = get_random_batch(BATCH_SIZE , X_train , y_train)
        x_batch_2, y_batch_2 = get_almost_identical_batch(BATCH_SIZE, X_train, y_train , y_batch_1)
        y_batch_1_label = np.nonzero(y_batch_1)[1]
        y_batch_2_label = np.nonzero(y_batch_2)[1]
        y_batch = np.equal(y_batch_1_label,y_batch_2_label).astype('float32')
        
        sess.run(train,
                 feed_dict = { network.x_1        : x_batch_1, 
                               network.x_2        : x_batch_2,
                               network.y          : y_batch, 
                               network.keep_prob  : DROPOUT_HOLD_PROB, 
                               network.is_training: True })

        if i%20 == 0:
            s = sess.run(summ,
                         feed_dict={network.x_1: x_batch_1,
                                    network.x_2: x_batch_2,
                                    network.y: y_batch,
                                    network.keep_prob: DROPOUT_HOLD_PROB,
                                    network.is_training: False})
            writer.add_summary(s, i)
            writer.flush()
        
        if i%100 == 0 :
            print ("###Step : %0d"  %i)
            
            train_loss = network.loss.eval( feed_dict = { network.x_1        : x_batch_1, 
                                                          network.x_2        : x_batch_2,
                                                          network.y          : y_batch, 
                                                          network.keep_prob  : 1,
                                                          network.is_training: True })
            train_loss_arr.append(train_loss)
            print("y_batch_1 is:\n%s" % y_batch_1_label)
            print("y_batch_2 is:\n%s" % y_batch_2_label)
            print("Train Loss is:\n%s" % train_loss)

            output_1  = network.output_1.eval( \
                                            feed_dict = { network.x_1        : x_batch_1,
                                                          network.x_2        : x_batch_2,
                                                          network.y          : y_batch,
                                                          network.keep_prob  : 1,
                                                          network.is_training: False })
            output_2  = network.output_2.eval( \
                                            feed_dict = { network.x_1        : x_batch_1,
                                                          network.x_2        : x_batch_2,
                                                          network.y          : y_batch,
                                                          network.keep_prob  : 1,
                                                          network.is_training: False })

            print("Output_embadings_1[0]:\n%s"% output_1[0])
            print("Output_embadings_2[0]:\n%s"% output_2[0])
            print("Output_embadings_1[1]:\n%s"% output_1[1])
            print("Output_embadings_2[1]:\n%s"% output_2[1])
            print('\n')

        if i % 200 == 0 :
            ####Graphics#####

            y_label = np.zeros([0])
            embedings_graphics = np.zeros([0,EMBEDINGS_SIZE])
            num_of_class_to_show = 10
            cmap = plt.cm.get_cmap('hsv', num_of_class_to_show)

            for b in range(20):
                x_batch_graphics, y_batch_graphics = get_random_batch(BATCH_SIZE, X_train, y_train)
                y_batch = np.ones([32,]).astype('float32')

                y_label_b = np.nonzero(y_batch_graphics)[1]

                embedings_graphics_b = network.output_1.eval( \
                                            feed_dict = { network.x_1        : x_batch_graphics,
                                                          network.x_2        : x_batch_graphics,
                                                          network.y          : y_batch,
                                                          network.keep_prob  : 1,
                                                          network.is_training: False })

                y_label = np.concatenate((y_label,y_label_b),axis=0)
                embedings_graphics = np.concatenate((embedings_graphics,embedings_graphics_b),axis=0)

            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
            two_d_embeddings = tsne.fit_transform(embedings_graphics)

            plt.figure(figsize=(10, 10))  # in inches
            for dot in range(y_label.shape[0]):
                x, y = two_d_embeddings[dot, :]
                if y_label[dot] < num_of_class_to_show :
                    plt.scatter(x, y,c=cmap(int(y_label[dot])))

            #plt.show()
            plt.title("TSNE embadings mapping after %0d steps"%i)
            plt.savefig("tsne_step_%0d.png"%i)

