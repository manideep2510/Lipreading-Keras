import glob
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score, accuracy_score
from funcsigs import signature

#from metrics import get_metrics

def plot_loss_and_acc(history, path):


    try:
        os.mkdir('/data/results/'+ path)
    except OSError:
        pass

    loss_train = []
    loss_val = []
    #acc_val = []
    #acc_train = []

    loss_train.append(history.history['loss'])
    loss_val.append(history.history['val_loss'])
    #acc_train.append(history.history['acc'])
    #acc_val.append(history.history['val_acc'])

    # Accuracy plots

    '''plt.plot(acc_train[0])
    plt.plot(acc_val[0])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.show()
    plt.savefig('/data/AV-speech-separation/results/' + path + '/accuracy.png')
    plt.close()

    print ('Saved Accuracy plot')'''

    # Loss plots

    plt.plot(loss_train[0])
    plt.plot(loss_val[0])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    #plt.show()
    plt.savefig('/data/results/' + path + '/loss.png')
    plt.close()

    print ('Saved Loss plot')
