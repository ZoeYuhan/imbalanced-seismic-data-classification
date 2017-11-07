import h5py
import tensorflow as tf
import CNN_series
import LSTM_FCN
import FCN
import res_net
import MLP
import data_preprocess
import data_pre
import data_vis
import os
import time
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, auc, roc_curve, precision_recall_curve, f1_score, \
    recall_score, precision_score
from sklearn.model_selection import StratifiedKFold, KFold
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from collections import Counter

NET = MLP
Th = 0.5
OUTPUT_NODE = 1
IMAGE_SIZE1 = 1
IMAGE_SIZE2 = 1222

BATCH_SIZE = 128
TRANING_STEPS = 20000

model_dir = "saver"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

MODEL_SAVE_PATH = model_dir
MODEL_NAME = "model.ckpt"


def model_train(train, valid, pos_weight):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None,
                                        IMAGE_SIZE1,
                                        IMAGE_SIZE2,
                                        NET.NUM_CHANNELS],
                           name='x-input')

        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
        phase = tf.placeholder(tf.bool, name='phase')
        y = NET.model(x, phase)

        global_step = tf.Variable(0, trainable=False)

        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=logit, targets=y_, pos_weight=pos_weight)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean
        correct_prediction = tf.equal(tf.greater(y, Th), tf.greater(y_, Th))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(0.00005).minimize(loss, global_step=global_step)

        with tf.control_dependencies([train_step]):
            train_op = tf.no_op(name='train')

        saver = tf.train.Saver()
        tf.summary.histogram('logits', tf.argmax(y, 1))
        tf.summary.histogram('labels', tf.argmax(y_, 1))
        tf.summary.histogram('loss', loss)
        tf.summary.histogram('accuracy', accuracy)

        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            summaries = tf.summary.merge_all()
            writer = tf.summary.FileWriter(
                os.path.join('./log_tensorboard', time.strftime("%Y-%m-%d-%H-%M-%S")))
            writer.add_graph(sess.graph)
            # Best validation accuracy seen so far.
            best_loss = 10000.0
            # Iteration-number for last improvement to validation accuracy.
            last_improvement = 0
            # Stop optimization if no improvement found in this many iterations.
            require_improvement = 1000
            for i in range(TRANING_STEPS):
                xs, ys = train.next_batch(BATCH_SIZE)
                reshaped_xs = np.reshape(xs, [BATCH_SIZE,
                                              IMAGE_SIZE1,
                                              IMAGE_SIZE2,
                                              NET.NUM_CHANNELS])
                reshaped_ys = np.reshape(ys, [BATCH_SIZE, OUTPUT_NODE])

                summ,_, loss_value, step = sess.run([summaries,train_op, loss, global_step],
                                               feed_dict={x: reshaped_xs, y_: reshaped_ys, phase: 1})
                writer.add_summary(summ, i)
                if i % 100 == 0 or (i == (TRANING_STEPS - 1)):
                    train_accuracy,logit = sess.run([accuracy, y], feed_dict={x: reshaped_xs, y_: reshaped_ys, phase: 1})
                    # print ("After %d training steps, loss %g, training accuracy %g" % (step, loss_value, train_accuracy))
                    if loss_value < best_loss:
                        # Update the best-known validation accuracy.
                        best_loss = loss_value
                        last_improvement = i+1
                        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                        improved_str = '*'
                    else:
                        improved_str = ''
                    msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Train-Loss: {2:>6.1%}, {3}"

                    # Print it.
                    print(msg.format(i + 1, train_accuracy,loss_value,  improved_str))

                    # If no improvement found in the required number of iterations.
                if i+1 - last_improvement > require_improvement:
                    print("No improvement found in a while, stopping optimization.")
                    break


            test_xs = np.reshape(valid.images, [-1,
                                                IMAGE_SIZE1,
                                                IMAGE_SIZE2,
                                                NET.NUM_CHANNELS])
            test_ys = np.reshape(valid.labels, [-1, OUTPUT_NODE])

            test_accuracy, test_logit = sess.run([accuracy, y], feed_dict={x: test_xs, y_: test_ys, phase: 1})
            a = test_logit.reshape([1, -1]) > Th
            a.astype("int")
            test_predict = a.reshape(-1)
            try:
                roc_score = roc_auc_score(valid.labels, test_predict)
            except ValueError:
                roc_score = 0
            print("-" * 75)
            print("After %d training steps, test accuracy %g" % (step, test_accuracy))

            # TODO:plot the confusion matrix
            target_names = ['normal', 'warning']
            cnf_matrix1 = confusion_matrix(valid.labels, test_predict)
            data_vis.plot_confusion_matrix(cnf_matrix1, classes=target_names,
                                           title='Confusion matrix')
            # TODO: plot the ROC Curves and AUC Score
            fpr, tpr, _ = roc_curve(valid.labels, test_predict, pos_label=1)
            auc_score = auc(fpr, tpr)
            fs = f1_score(valid.labels, test_predict)
            G_mean = np.sqrt(np.mean(tpr) * np.mean(fpr))
            data_vis.plot_roc_curve(fpr, tpr, auc_score)
            # TODO: plot Precision and Recall Curves
            precision, recall, _ = precision_recall_curve(valid.labels, test_predict, pos_label=1)
            auc_score_1 = auc(recall, precision)
            r = recall_score(valid.labels, test_predict)
            p = precision_score(valid.labels, test_predict)
            data_vis.plot_precision_recall_curve(recall, precision, auc_score_1)

            print("ROC AUC : %.10f" % auc_score)
            print("G_mean : %.10f" % G_mean)
            print("Sensitivity(TPR) : %.10f" % np.mean(tpr))
            print("Recall : %.10f" % r)
            print("Precision: %.10f" % p)
            print("F1 Score : %.10f" % fs)
            print("Specificity(TNR) : %.10f" % (1 - np.mean(fpr)))
            print("PR AUC : %.10f" % (auc_score_1))

            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


if __name__ == "__main__":
    st = time.time()
    file = h5py.File('features_combine.h5', 'r')
    data = file['train_features'][:]
    label = file['train_labels'][:]
    X_test = file['test_features'][:]
    Y_test = file['test_labels'][:]
    file.close()

    nb_folds = 10

    kfolds = StratifiedKFold(n_splits=nb_folds, shuffle=True, random_state=None)
    kfolds.get_n_splits(data, label)
    st = time.time()
    for train, valid in kfolds.split(data, label):
        X, y = np.array(data[train]), np.array(label[train])
        # ada = SMOTE(random_state=42)
        # X_res, y_res = ada.fit_sample(X, y)
        # print('Resampled dataset shape {}'.format(Counter(y_res)))
        # train_input = data_preprocess.DataSet(np.array(X_res), np.array(y_res))
        # pw = np.sum(y_res == 0) / np.sum(y_res == 1)
        print('dataset shape {}'.format(Counter(y)))
        train_input=data_preprocess.DataSet(np.array(X),np.array(y))
        pw = np.sum(y == 0) / np.sum(y == 1)
        valid_input = data_preprocess.DataSet(np.array(data[valid]), np.array(label[valid]))
        print('positive weight:', pw)
        model_train(train_input, valid_input, pos_weight=pw)
        print("*" * 75)

    end = time.time()


