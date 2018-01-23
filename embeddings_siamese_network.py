
#!/usr/bin/env python

"""
    File name:          embeddings_siamese_network.py
    Author:             Guy Tevet & Chen Ponchek
    Date created:       23/1/2018
    Date last modified: 23/1/2018
    Description:        siamese network for embeddings application
"""


import tensorflow as tf
import numpy as np


# In[2]:


from clasification_cnn_network import base_network


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
FC_OUTPUT_SIZE          = 16
BATCH_SIZE              = 32
DROPOUT_HOLD_PROB       = 0.9
LEARNING_RATE           = 0.0001
LOSS_MARGIN             = 1.0


# In[4]:


class siamese_network(base_network) :
    
    def __init__(self , arch):
        self.arch = arch
        self.x_1         = tf.placeholder(tf.float32,name='x_1',shape=[BATCH_SIZE,INPUT_NUM_ROWS,INPUT_NUM_COLUMNS,INPUT_CHANNELS])
        self.x_2         = tf.placeholder(tf.float32,name='x_2',shape=[BATCH_SIZE,INPUT_NUM_ROWS,INPUT_NUM_COLUMNS,INPUT_CHANNELS])
        self.y           = tf.placeholder(tf.float32,name='y',shape=BATCH_SIZE)
        self.keep_prob   = tf.placeholder(tf.float32,name='keep_prob')
        self.is_training = tf.placeholder(tf.bool,name='is_training')

        self.output_size = FC_OUTPUT_SIZE
        
        with tf.variable_scope("siamese_network") as scope:
            self.output_1 = self.network(self.x_1)
            scope.reuse_variables()
            self.output_2 = self.network(self.x_2)
            
        self.loss = self.contrastive_loss()

    def loss_with_spring(self):
        margin = LOSS_MARGIN
        labels_t = self.y
        labels_f = tf.subtract(1.0, self.y, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.output_1, self.output_2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss
    
    def contrastive_loss(self):
        D = tf.pow(tf.norm(tf.subtract(self.output_1, self.output_2),ord='euclidean',name='D',axis=1),2)
        same_label_loss = tf.multiply(tf.multiply(0.5,D),self.y,name='same_label_loss')
        diff_label_loss = tf.multiply(tf.multiply(0.5,tf.subtract(1.0, self.y)),tf.maximum(0.0,tf.subtract(tf.multiply(LOSS_MARGIN , tf.ones(BATCH_SIZE)),D)),name='diff_label_loss')
        loss = tf.add(same_label_loss,diff_label_loss,name='contrastive_loss')
        avg_loss = tf.reduce_mean(loss)
        tf.summary.scalar("embeddings_loss", avg_loss)

        return loss

