
#!/usr/bin/env python

"""
    File name:          clasification_cnn_network.py
    Author:             Guy Tevet & Chen Ponchek
    Date created:       23/1/2018
    Date last modified: 23/1/2018
    Description:        base network class & arch configurations class
"""


import tensorflow as tf
import numpy as np


# In[2]:


NUM_CLASSES             = 46
INPUT_NUM_ROWS          = 256
INPUT_NUM_COLUMNS       = 96
INPUT_CHANNELS          = 1
CONV_BLOCK_KERNEL_SIZE        = 5
CONV_FIRST_KERNEL_SIZE        = 10
CONV1_OUTPUT_CHANNELS   = 64
CONV2_1_OUTPUT_CHANNELS = 64
CONV2_2_OUTPUT_CHANNELS = 128
RESNEXT_OUTPUT_CHANNELS = 128
NUM_HIDDEN_LAYERS       = 1024
FC_OUTPUT_SIZE          = NUM_CLASSES
DROPOUT_HOLD_PROB       = 0.9
LEARNING_RATE           = 0.0001

#AGGREGATION_METHOD      = 'global_aggregation'
#AGGREGATION_METHOD      = 'avg_pooling'
#AGGREGATION_METHOD      = 'max_pooling'

#NETWORK_ARCH            = 'vanilla_cnn'
#NETWORK_ARCH            = 'resNeXt'
#DEEP_ARCH               = True


# In[3]:

class network_arch:
    def __init__(self,type,deep,aggregation_method):
        self.type = type
        self.deep = deep
        self.aggregation_method = aggregation_method

class base_network:

    def __init__(self,arch):
        self.arch = arch
        self.x = tf.placeholder(tf.float32,shape=[None,INPUT_NUM_ROWS,INPUT_NUM_COLUMNS,INPUT_CHANNELS])
        self.y_true = tf.placeholder(tf.float32,shape=[None,NUM_CLASSES])
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)

        self.output_size = FC_OUTPUT_SIZE

        self.y_pred = self.network(self.x)
        self.loss = self.cross_entropy_loss()

        
    def network(self,x):



        if (self.arch.type == 'vanilla_cnn' and not self.arch.deep):
            print('selected network architecture to be shallow vanilla CNN')
            CONV1_OUTPUT_CHANNELS = 64

            convo_1 = self.convolutional_layer('conv_base', x, shape=[CONV_FIRST_KERNEL_SIZE,CONV_FIRST_KERNEL_SIZE,INPUT_CHANNELS,CONV1_OUTPUT_CHANNELS])
            convo_1 = tf.nn.max_pool(convo_1, [1, 2, 2, 1], [1, 2, 2, 1],padding='SAME')

            convo_block_1 = self.convolutional_block('conv_block_0',convo_1,CONV_BLOCK_KERNEL_SIZE,CONV2_1_OUTPUT_CHANNELS,CONV2_2_OUTPUT_CHANNELS)
            convo_block_1 = tf.nn.max_pool(convo_block_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            conv_done = convo_block_1
        elif (self.arch.type == 'vanilla_cnn' and self.arch.deep):
            print('selected network architecture to be deep vanilla CNN')
            CONV_BASE_OUT_CH = 16
            CONV_BLOCK_OUT_CH = [32,32,32,64,64,64]
            CONV_BLOCK_NUM = len(CONV_BLOCK_OUT_CH)

            conv_base = self.convolutional_layer('conv_base', x, shape=[CONV_FIRST_KERNEL_SIZE, CONV_FIRST_KERNEL_SIZE, INPUT_CHANNELS,CONV_BASE_OUT_CH])
            conv_base = tf.nn.max_pool(conv_base, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            conv_block = [None] * CONV_BLOCK_NUM
            for block_i in range(CONV_BLOCK_NUM):
                if block_i==0:
                    conv_block[block_i] = self.convolutional_block("conv_block_%0d"%block_i,conv_base,
                                                             CONV_BLOCK_KERNEL_SIZE,CONV_BASE_OUT_CH,CONV_BLOCK_OUT_CH[block_i])
                    conv_block[block_i] = tf.nn.max_pool(conv_block[block_i], [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
                else:
                    conv_block[block_i] = self.convolutional_block("conv_block_%0d"%block_i, conv_block[block_i-1],
                                                             CONV_BLOCK_KERNEL_SIZE, CONV_BLOCK_OUT_CH[block_i-1], CONV_BLOCK_OUT_CH[block_i])

            conv_done = conv_block[CONV_BLOCK_NUM - 1]


        elif (self.arch.type == 'resNeXt' and not self.arch.deep):
            print('selected network architecture to be shallow ResNeXt')
            CONV1_OUTPUT_CHANNELS = 128
            convo_1 = self.convolutional_layer('convo_1', x, shape=[CONV_FIRST_KERNEL_SIZE,CONV_FIRST_KERNEL_SIZE,INPUT_CHANNELS,CONV1_OUTPUT_CHANNELS])
            convo_1 = tf.nn.max_pool(convo_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            resnext_block_1 = self.resnext_block('resnext_block_1',convo_1,RESNEXT_OUTPUT_CHANNELS)
            resnext_block_1 = tf.nn.max_pool(resnext_block_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            conv_done = resnext_block_1

        elif (self.arch.type == 'resNeXt' and self.arch.deep):
            print('selected network architecture to be deep ResNeXt')
            CONV_BASE_OUT_CH = 32
            RESNEXT_BLOCK_OUT_CH = [64,64,64,128,128,128]
            RESNEXT_BLOCK_NUM = len(RESNEXT_BLOCK_OUT_CH)

            conv_base = self.convolutional_layer('conv_base', x, shape=[CONV_FIRST_KERNEL_SIZE, CONV_FIRST_KERNEL_SIZE, INPUT_CHANNELS,CONV_BASE_OUT_CH])
            conv_base = tf.nn.max_pool(conv_base, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            resnext_block = [None] * RESNEXT_BLOCK_NUM
            for block_i in range(RESNEXT_BLOCK_NUM):
                if block_i==0:
                    resnext_block[block_i] = self.resnext_block("resnext_block_%0d"%block_i,conv_base,RESNEXT_BLOCK_OUT_CH[block_i])
                    resnext_block[block_i] = tf.nn.max_pool(resnext_block[block_i], [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
                else:
                    resnext_block[block_i] = self.resnext_block("resnext_block_%0d"%block_i, resnext_block[block_i-1], RESNEXT_BLOCK_OUT_CH[block_i])

            conv_done = resnext_block[RESNEXT_BLOCK_NUM - 1]


        #reshaping
        pre_aggregation_reshape = tf.reshape(conv_done,[-1,conv_done.shape[1],conv_done.shape[2]*conv_done.shape[3]]) #DIMS [BATCH x T x F*C]

        if (self.arch.aggregation_method == 'global_aggregation'):
            print('selected aggregation method to be global_aggregation')
            aggregated = self.global_aggregation_layer('global_aggregation', pre_aggregation_reshape)
        elif (self.arch.aggregation_method == 'max_pooling'):
            print('selected aggregation method to be max_pooling')
            aggregated = tf.squeeze(tf.squeeze(tf.nn.max_pool(
                tf.expand_dims(pre_aggregation_reshape,axis=3), [1,1,1,1],[1,pre_aggregation_reshape.shape[1],1,1],padding='SAME'),axis=3),axis=1)
        elif (self.arch.aggregation_method == 'avg_pooling'):
            print('selected aggregation method to be avg_pooling')
            aggregated = tf.squeeze(tf.squeeze(tf.nn.avg_pool(
                tf.expand_dims(pre_aggregation_reshape,axis=3), [1,1,1,1],[1,pre_aggregation_reshape.shape[1],1,1],padding='SAME'),axis=3),axis=1)


        fc_1 = self.fully_connected_layer('fc_1', aggregated, NUM_HIDDEN_LAYERS)

        fc_2 = self.fully_connected_layer('fc_2', fc_1, NUM_HIDDEN_LAYERS)

        fc_3 = self.fully_connected_layer('fc_3', fc_2, NUM_HIDDEN_LAYERS)

        y_pred = self.fully_connected_layer('fc_4', fc_3, self.output_size)
        return y_pred
    
    def convolutional_layer(self, name, x, shape):
        with tf.name_scope(name):
            initer = tf.truncated_normal_initializer(stddev=0.01)
            W = tf.get_variable(name+'W', dtype=tf.float32, shape=shape, initializer=initer)
            b = tf.get_variable(name+'b', 
                                dtype=tf.float32, 
                                initializer=tf.constant(0.01, shape=[shape[3]], dtype=tf.float32))
            act = tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b)
            
            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)

            return act
    

    def convolutional_block(self, name, x, kernal_size, in_block_output_size, block_output_size):
        with tf.name_scope(name):
            input_channels = x.get_shape()[3]
            convo_1 = self.convolutional_layer(name+'_1', x,       shape=[kernal_size,kernal_size,input_channels,in_block_output_size])
            convo_2 = self.convolutional_layer(name+'_2', convo_1, shape=[kernal_size,kernal_size,in_block_output_size,block_output_size])
            return convo_2

        
    def global_aggregation_layer(self, name, input_layer):
        with tf.name_scope(name):
            initer = tf.truncated_normal_initializer(stddev=0.01)
            W = tf.get_variable(name+'W', dtype=tf.float32, shape=[int(input_layer.get_shape()[2]),1], initializer=initer)

            input_layer_reshape = tf.reshape(input_layer, [-1 , input_layer.get_shape()[2]]) # DIMS [BATCH_SIZE * T x F * C]

            weights = tf.nn.softmax(tf.nn.tanh(tf.matmul(input_layer_reshape,W))) # DIMS [ BATCH_SIZE * T x 1] #FIXME: return b
            weights = tf.reshape(weights , [input_layer.get_shape()[1] , -1] ) #DIMS [T x BATCH_SIZE]

            weights_tiled = tf.expand_dims(weights,axis=2)
            weights_tiled = tf.tile(weights_tiled,[1,1,input_layer.get_shape()[2]]) #DIMS [T x BATCH_SIZE x F * C]
            weights_tiled = tf.transpose(weights_tiled,[1,0,2]) #DIMS [BATCH x T x F * C]

            global_aggr_res = tf.multiply(input_layer , weights_tiled)
            global_aggr_res = tf.reduce_sum(global_aggr_res , axis= 1)

            tf.summary.histogram("weights", weights)
            tf.summary.histogram("global_aggr_res", global_aggr_res)

            return global_aggr_res
        
    def fully_connected_layer(self, name, input_layer, size):
        with tf.name_scope(name):
            n_prev_weight = input_layer.get_shape()[1]
            initer = tf.truncated_normal_initializer(stddev=0.01)
            W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, size], initializer=initer)
            b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[size], dtype=tf.float32))
            fc = tf.nn.bias_add(tf.matmul(input_layer, W), b)
            fc_bn = tf.layers.batch_normalization(fc, training=self.is_training, reuse=tf.AUTO_REUSE, name=name + 'BN')
            fc_drop = tf.nn.dropout(fc_bn,keep_prob=self.keep_prob)
            act = tf.nn.relu(fc_drop)
            
            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)

            return act

   
    def resnext_block(self, name, input_layer, output_channel):
        '''
        The block structure in Figure 3b. Takes a 4D tensor as input layer and splits, concatenates
        the tensor and restores the depth. Finally adds the identity and ReLu.
        :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
        input_channel]
        :param output_channel: int, the number of channels of the output
        :return: 4D tensor in shape of [batch_size, output_height, output_width, output_channel]
        '''
        with tf.name_scope(name):
            input_channel = input_layer.get_shape().as_list()[-1]

            # When it's time to "shrink" the image size, we use stride = 2
            if input_channel * 2 == output_channel:
                increase_dim = True
                stride = 2
            elif input_channel == output_channel:
                increase_dim = False
                stride = 1
            else:
                raise ValueError('Output and input channel does not match in residual blocks!!!')

            concat_bottleneck = self.bottleneck_b(name+'_bottleneck_b', input_layer, stride)

            bottleneck_depth = concat_bottleneck.get_shape().as_list()[-1]

            # Restore the dimension. Without relu here
            restore = self.conv_bn_relu_layer(name+'_conv_bn_relu_layer', input_layer=concat_bottleneck,
                                              filter_shape=[1, 1, bottleneck_depth, output_channel],
                                              stride=1, relu=False)

            # When the channels of input layer and conv2 does not match, we add zero pads to increase the
            #  depth of input layers
            if increase_dim is True:
                pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                              strides=[1, 2, 2, 1], padding='VALID')
                padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                              input_channel // 2]])
            else:
                padded_input = input_layer

            # According to section 4 of the paper, relu is played after adding the identity.
            output = tf.nn.relu(restore + padded_input)

            #output_reshape = tf.reshape(output,[-1,INPUT_NUM_ROWS,INPUT_NUM_COLUMNS*RESNEXT_OUTPUT_CHANNELS]) #DIMS [BATCH x T x F*C]
            return output


    def bottleneck_b(self, name, input_layer, stride):
        '''
        The bottleneck strucutre in Figure 3b. Concatenates all the splits
        :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
        input_channel]
        :param stride: int. 1 or 2. If want to shrink the image size, then stride = 2
        :return: 4D tensor in shape of [batch_size, output_height, output_width, output_channel]
        '''
        split_list = []
        for i in range(1): #FLAGS.cardinality
            with tf.variable_scope('split_%i'%i):
                splits = self.split(name+'split_%i'%i, input_layer=input_layer, stride=stride)
            split_list.append(splits)

        # Concatenate splits and check the dimension
        concat_bottleneck = tf.concat(values=split_list, axis=3, name='concat')

        return concat_bottleneck


    def conv_bn_relu_layer(self, name, input_layer, filter_shape, stride, relu=True):
        '''
        A helper function to conv, batch normalize and relu the input tensor sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :param relu: boolean. Relu after BN?
        :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
        '''

        out_channel = filter_shape[-1]
        filter = self.create_variables(name=name+'_conv', shape=filter_shape)

        conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        bn_layer = self.batch_normalization_layer(name+'_batch_normalization_layer',conv_layer, out_channel)

        if relu is True:
            output = tf.nn.relu(bn_layer)
        else:
            output = bn_layer
        return output

    def create_variables(self, name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
        '''
        Create a variable with tf.get_variable()
        :param name: A string. The name of the new variable
        :param shape: A list of dimensions
        :param initializer: User Xavier as default.
        :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
        layers.
        :return: The created variable
        '''

        ## TODO: to allow different weight decay to fully connected layer and conv layer
        if is_fc_layer is True:
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.95) #FLAGS.weight_decay
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.95) #FLAGS.weight_decay

        new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
        return new_variables


    def batch_normalization_layer(self, name, input_layer, dimension):
        '''
        Helper function to do batch normalziation
        :param input_layer: 4D tensor
        :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
        :return: the 4D tensor after being normalized
        '''
        with tf.name_scope(name):
            mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
            beta = tf.get_variable(name+'_beta', dimension, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(name+'_gamma', dimension, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))
            bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, 0.001)

            return bn_layer
    
    def split(self, name, input_layer, stride):
        '''
        The split structure in Figure 3b of the paper. It takes an input tensor. Conv it by [1, 1,
        64] filter, and then conv the result by [3, 3, 64]. Return the
        final resulted tensor, which is in shape of [batch_size, input_height, input_width, 64]
        :param input_layer: 4D tensor in shape of [batch_size, input_height, input_width,
        input_channel]
        :param stride: int. 1 or 2. If want to shrink the image size, then stride = 2
        :return: 4D tensor in shape of [batch_size, input_height, input_width, input_channel/64]
        '''

        input_channel = input_layer.get_shape().as_list()[-1]
        num_filter = 64 #FLAGS.block_unit_depth
        # according to Figure 7, they used 64 as # filters for all cifar10 task

        with tf.variable_scope('bneck_reduce_size'):
            conv = self.conv_bn_relu_layer(name, input_layer, filter_shape=[1, 1, input_channel, num_filter],
                                  stride=stride)
        with tf.variable_scope('bneck_conv'):
            conv = self.conv_bn_relu_layer(name, conv, filter_shape=[3, 3, num_filter, num_filter], stride=1)

        return conv

    def cross_entropy_loss(self):
        with tf.name_scope('cross_entropy_loss'):
            trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            regularizer = tf.contrib.layers.l2_regularizer(scale=1e-6)
            reg_term = tf.contrib.layers.apply_regularization(regularizer, trainable_variables)

            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true,logits=self.y_pred))
            loss = cross_entropy + reg_term
            
            tf.summary.scalar("reg_term_loss", reg_term)
            tf.summary.scalar("cross_entropy_loss", cross_entropy)
            tf.summary.scalar("loss", loss)
            return loss

