#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 8/8/17 PM10:37
# @Author : Zoe
# @Site : 
# @File : CNN_series.py
# @Software: PyCharm Community Edition

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.training.moving_averages import assign_moving_average
import  numpy as np
NUM_CHANNELS=1
# NUM_LABELS=10

CONV1_DEEP=32
CONV1_SIZE_row=1
CONV1_SIZE_col=3

CONV2_DEEP=32
CONV2_SIZE_row=1
CONV2_SIZE_col=3

CONV3_DEEP=64
CONV3_SIZE_row=1
CONV3_SIZE_col=3

FC_SIZE=1024

OUTPUT_NODE=1

# LSTM hyperparameter
n_hidden_unins=8
n_inputs=1
n_steps=541

def model(input_tensor,phase):
    # print(input_tensor.get_shape())
#LSTM network
    with tf.variable_scope('LSTM-layer'):
        lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden_unins)
        output, intermediate_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=tf.reshape(input_tensor,[-1,n_steps,n_inputs]),dtype=tf.float32)
    # print (output.get_shape())#(?, 18, 8)
#FCN network
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE_row, CONV1_SIZE_col, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP],
                                       initializer=tf.constant_initializer(0.1))

        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")

        bn_fc1 = tf.layers.batch_normalization(conv1, training=phase)
        relu1 = tf.nn.relu(tf.nn.bias_add(bn_fc1,conv1_biases))

    with tf.variable_scope('layer2-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE_row, CONV2_SIZE_col, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.1))

        conv2 = tf.nn.conv2d(relu1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")

        bn_fc2 = tf.layers.batch_normalization(conv2, training=phase)
        relu2 = tf.nn.relu(tf.nn.bias_add(bn_fc2,conv2_biases))

    with tf.variable_scope('layer3-conv3'):
        conv3_weights = tf.get_variable("weight", [CONV3_SIZE_row, CONV3_SIZE_col, CONV2_DEEP, CONV3_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.1))

        conv3 = tf.nn.conv2d(relu2, conv3_weights, strides=[1, 1, 1, 1], padding="SAME")

        bn_fc3 = tf.layers.batch_normalization(conv3, training=phase)
        relu3 = tf.nn.relu(tf.nn.bias_add(bn_fc3,conv3_biases))

    with tf.variable_scope('layer4-pool'):
        # fc_weights = tf.get_variable("weights", [CONV3_DEEP, OUTPUT_NODE],
        #                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        # fc_biases = tf.get_variable("bias", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        globalpool = tf.nn.avg_pool(relu3, ksize=[1, 1, 18, 1], strides=[1, 1, 18, 1], padding="SAME")
        # reshaped = tf.reshape(globalpool, [-1, CONV3_DEEP])
        # y = tf.nn.sigmoid(tf.matmul(reshaped, fc_weights) + fc_biases)
        # print (globalpool.get_shape())  #(?, 1, 1, 64)

    with tf.variable_scope('output'):
        lstm_shape = output.get_shape().as_list()
        fcn_shape=globalpool.get_shape().as_list()
        x=tf.reshape(output,[-1,lstm_shape[1]*lstm_shape[2]])
        y=tf.reshape(globalpool,[-1,fcn_shape[1]*fcn_shape[2]*fcn_shape[3]])
        # print(x.get_shape())#(?, 144)
        # print(y.get_shape())#(?, 64)
        reshaped=tf.concat([x,y],axis=1)
        # print (reshaped.get_shape())#(?, 208)
        fc_weights = tf.get_variable("weights", [reshaped.get_shape()[1], OUTPUT_NODE],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc_biases = tf.get_variable("bias", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        logit= tf.nn.sigmoid(tf.matmul(reshaped, fc_weights) + fc_biases)
    return logit


