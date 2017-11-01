#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/10 15:39
# @Author  : Zoe
# @Site    : 
# @File    : data_vis.py
# @Software: PyCharm Community Edition
import numpy as np
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(1)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def plot_roc_curve(fpr,tpr,auc_score):
    plt.figure(2)
    plt.title('ROC Curves and AUC Score')
    plt.plot(fpr, tpr, '-', linewidth=1, label="AUC Score: %f" % (auc_score))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.show()

def plot_precision_recall_curve(recall,precision,auc_score):
    plt.figure(3)
    plt.title('Precision and Recall Curves')
    plt.plot(recall,precision, '-', linewidth=1, label="AUC Score: %f" % (auc_score))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    from sklearn.decomposition import PCA
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import SMOTE
   
    csv = np.genfromtxt('data.csv', delimiter=",")
    y = np.array(csv[:, -1])
    X = np.array(csv[:, :-1])

    # Instanciate a PCA object for the sake of easy visualisation
    pca = PCA(n_components=2)
    # Fit and transform x to visualise inside a 2D feature space
    X_vis = pca.fit_transform(X)
    # Apply SMOTE + ENN
    sm = SMOTEENN()
    X_resampled, y_resampled = sm.fit_sample(X, y)
    X_res_vis = pca.transform(X_resampled)

    # Two subplots, unpack the axes array immediately
    f, (ax1, ax2) = plt.subplots(1, 2)

    c0 = ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0",
                     alpha=0.5)
    c1 = ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1",
                     alpha=0.5)
    ax1.set_title('Original set')

    ax2.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1],
                label="Class #0", alpha=0.5)
    ax2.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1],
                label="Class #1", alpha=0.5)
    ax2.set_title('SMOTE + ENN')

    # make nice plotting
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.set_xlim([-6, 15])
        ax.set_ylim([-6, 10])

    f.legend((c0, c1), ('Class #0', 'Class #1'), loc='lower center',
             ncol=2, labelspacing=0.)
    plt.tight_layout(pad=3)
    plt.show()
