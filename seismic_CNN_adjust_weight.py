# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 09:07:44 2017

@author: lingling.su
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 8/8/17 PM10:53
# @Author : Zoe
# @Site :
# @File : CNN_Series_train.py
# @Software: PyCharm Community Edition


import tensorflow as tf
import CNN_series
import LSTM_FCN
import FCN_1 as FCN
import res_net
import MLP
import data_preprocess

import data_vis
import os
import numpy as np
import h5py
import pandas as pd
#from sklearn import preprocessing
#from tensorflow.python.framework import ops
#from tensorflow.python.framework import dtypes
from sklearn.metrics import roc_auc_score,confusion_matrix,auc,roc_curve,precision_recall_curve,f1_score,recall_score,precision_score
from sklearn.model_selection import StratifiedKFold,KFold
from imblearn.over_sampling import ADASYN,SMOTE,RandomOverSampler
import time
from collections import Counter

NET=FCN
OUTPUT_NODE=1
IMAGE_SIZE1=1
# IMAGE_SIZE2=528

BATCH_SIZE=16
TRANING_STEPS=2000

THRESH=0.5
th=0.5

model_dir = "saver"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

MODEL_SAVE_PATH=model_dir
MODEL_NAME="model.ckpt"

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True


def model_train(train,valid, pos_weight, IMAGE_SIZE2):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None,
                                        IMAGE_SIZE1,
                                        IMAGE_SIZE2,
                                        NET.NUM_CHANNELS],
                           name='x-input')

        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
        phase = tf.placeholder(tf.bool, name='phase')
        weight = tf.placeholder(tf.float32, [None,OUTPUT_NODE], name='weight')
        logit, y = NET.model(x,phase)

        global_step = tf.Variable(0, trainable=False)

        cross_entropy = pow((tf.subtract(y_, weight*y)),2)
        # cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=weight*logit, targets=y_, pos_weight=1)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean
        correct_prediction = tf.equal(tf.greater(y, THRESH), tf.greater(y_, THRESH))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_step = tf.train.AdamOptimizer(0.0005).minimize(loss, global_step=global_step)


        with tf.control_dependencies([train_step]):
            train_op=tf.no_op(name='train')


        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            weight_adjust=1
            for i in range(TRANING_STEPS):
                xs, ys = train.next_batch(BATCH_SIZE)
                palette=[0,1,2]
                key=np.array([1,weight_adjust])
                index=np.digitize(ys.ravel(),palette,right=True)
                print ('ys',ys.ravel(),ys)
                k_1=key[index]
                print ("index",index)
                k=key[index].reshape([-1,OUTPUT_NODE])
                print ("K is :",k.reshape([-1]))
                reshaped_xs = np.reshape(xs, [BATCH_SIZE,
                                              IMAGE_SIZE1,
                                              IMAGE_SIZE2,
                                              NET.NUM_CHANNELS])

                reshaped_ys=np.reshape(ys,[BATCH_SIZE,OUTPUT_NODE])



                _, loss_value, step, train_logit= sess.run([train_op, loss, global_step,y], feed_dict={x: reshaped_xs,
                                                                                                       y_: reshaped_ys,
                                                                                                       weight:k,
                                                                                                       phase:1})

                train_accuracy,_ = sess.run([accuracy, y], feed_dict={x: reshaped_xs, y_: reshaped_ys, phase: 1})
                print ("After %d training steps, loss %g, training accuracy %g" % (step, loss_value, train_accuracy))

                a = train_logit.reshape([1, -1]) > th

                a=a.astype("int")

                train_predict=a.reshape(-1)

                cm =confusion_matrix(ys, train_predict)
                print(ys,train_predict)
#                print (ys,train_predict,cm)
                try:
                    tp,tn,fp,fn=cm[0,0],cm[0,1],cm[1,0],cm[1,1]
                except IndexError:
                    tp,tn,fp,fn=cm[0,0],0,0,0
                fpr, tpr, thresh = roc_curve(ys, train_logit, pos_label=1)
                print( fpr, tpr, thresh)
                acc=(tp+tn)/(tp+tn+fp+fn)

                rauc = auc(fpr, tpr)
                tpr = (0 if np.isnan(np.mean(tpr)) else np.mean(tpr))
                fpr = (0 if np.isnan(np.mean(fpr)) else np.mean(fpr))
                gmean=np.sqrt(tpr*fpr)
                rauc=(0 if np.isnan(rauc) else rauc)
                counts=Counter(ys)
                H=counts[0]/(counts[1]+0.001)
                weight_adjust=H*np.exp(-gmean/2)*np.exp(-acc/2)
                # weight_adjust = H * np.exp(-rauc / 2)

                print ("weight adjust:",weight_adjust)
                print ("H:",H)
                print ("g_mean%g,exp(g_mean):", gmean,np.exp(-gmean/2))
                print("auc%g,exp(auc):", rauc, np.exp(-rauc / 2))
                print("acc%g,exp(acc)%g:", acc,np.exp(-acc/2))

#TODO: Predict the test
            test_xs = np.reshape(valid.images, [-1,
                                            IMAGE_SIZE1,
                                            IMAGE_SIZE2,
                                            NET.NUM_CHANNELS])
            test_ys = np.reshape(valid.labels, [-1, OUTPUT_NODE])

            test_accuracy, test_logit = sess.run([accuracy, y], feed_dict={x: test_xs, y_: test_ys,phase:1})
            a = test_logit.reshape([1, -1]) > THRESH
            a=a.astype("int")
            test_predict=a.reshape(-1)
            try:
                roc_score=roc_auc_score( valid.labels,test_predict)
            except ValueError:
                roc_score=0
            print ("-"*75)
            print ("After %d training steps, test accuracy %g, roc_score %g" % (step, test_accuracy,roc_score))
#
##TODO:plot the confusion matrix
            target_names = ['normal', 'warning']
            cnf_matrix1 = confusion_matrix(valid.labels, test_predict)
            data_vis.plot_confusion_matrix(cnf_matrix1, classes=target_names,
                          title='Confusion matrix')
##TODO: plot the ROC Curves and AUC Score
            fpr, tpr, thresh = roc_curve(valid.labels, test_predict, pos_label=1)
            fs=f1_score(valid.labels, test_predict)
            G_mean=np.sqrt(np.mean(tpr)*np.mean(fpr))
            auc_score=auc(fpr, tpr)
            data_vis.plot_roc_curve(fpr,tpr,auc_score)
##TODO: plot Precision and Recall Curves
            precision, recall, thresh1 = precision_recall_curve(valid.labels, test_predict, pos_label=1)
            auc_score_1=auc(recall,precision)
            r=recall_score(valid.labels,test_predict)
            p=precision_score(valid.labels,test_predict)
            data_vis.plot_precision_recall_curve(recall,precision,auc_score_1)

            acc_score = (cnf_matrix1[0,0]+cnf_matrix1[1,1])/(np.sum(cnf_matrix1))
            print("Accuracy: %.10f" % acc_score)
            print("ROC AUC : %.10f" % auc_score)
            print("PR AUC : %.10f" % (auc_score_1))
            print("G_mean : %.10f" % G_mean)
            print("Sensitivity(TPR) : %.10f" % np.mean(tpr))
            print("Recall : %.10f" % r)
            print("Precision: %.10f" % p)
            print("F1 Score : %.10f" % fs)
            print("Specificity(TNR) : %.10f" % (1 - np.mean(fpr)))
            print ("thresh is %s" %np.mean(thresh),"thresh1 is %s"%np.mean(thresh1))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

if __name__=="__main__":

#    label = np.genfromtxt ('F:\\PycharmProjects\\github\\CNN_series\\data\\trainingData\\trainingLabels1.csv', delimiter=',')
#    data = np.genfromtxt ('F:\\PycharmProjects\\github\\CNN_series\\data\\trainingData\\trainingData.csv', delimiter=',')
    file = h5py.File('./data/Pre_data.h5', 'r')
    data = file['train_data'][:]
    label = file['train_label'][:].reshape([-1])
#    X_test = file['test_features'][:]
#    Y_test = file['test_labels'][:]
    file.close()
    _, IMAGE_SIZE2 = data.shape

    nb_folds=10
    kfolds = StratifiedKFold(n_splits=nb_folds,shuffle=True,random_state=None)
    kfolds.get_n_splits(data, label)
    st = time.time()
    cv = 0
    cnf_matrix1_list,fpr_list,tpr_list,auc_score_list,precision_list,recall_list,auc_score_1_list=[],[],[],[],[],[],[]
    for train, valid in kfolds.split(data,label):
        st1 = time.time()
        cv = cv + 1
        print("{} cross validation!".format(cv))
        X,y=np.array(data[train]), np.array(label[train])
        ada = SMOTE(random_state=42)
#        H=Counter(y)[0]/Counter(y)[1]
#        print (H,Counter(y))
        X_res, y_res = ada.fit_sample(X,y)
        print('Resampled dataset shape {}'.format(Counter(y_res)))
        train_input=data_preprocess.DataSet(np.array(X),np.array(y))
        print('dataset shape {}'.format(Counter(y)))
        # train_input=data_preprocess.DataSet(np.array(X_res),np.array(y_res))
        valid_input=data_preprocess.DataSet(np.array(data[valid]),np.array(label[valid]))
        print('Valid dataset shape {}'.format(Counter(label[valid])))
        model_train(train_input, valid_input, pos_weight=1, IMAGE_SIZE2=IMAGE_SIZE2)
        end1 = time.time()
        print("{} cross validation time spend is: {}s".format(cv, (end1 - st1)))

        print("*"*75)
