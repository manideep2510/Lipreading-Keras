from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
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
import sys
from io import StringIO

class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """

    def __init__(self):
        Callback.__init__(self)


    def on_epoch_end(self, epoch, logs={}):
        msg = "Epoch: %i, %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.iteritems()))
        print(msg)
        
#loggingcallback = LoggingCallback(

def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.1
    epochs_drop = 10
    lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
    return lrate

def learningratescheduler():
    learningratescheduler = LearningRateScheduler(step_decay)
    return learningratescheduler

def earlystopping():
    earlystopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    return earlystopping

def reducelronplateau():
    reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr = 0.00000001)
    return reducelronplateau

    def __init__(self, logsdir):
        self.terminal = sys.stdout
        self.log = open(os.path.join(logsdir, 'log.txt'), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class LearningRateSchedulerPerBatch(LearningRateScheduler):
    """ Callback class to modify the default learning rate scheduler to operate each batch"""
    def __init__(self, schedule, verbose=0):
        super(LearningRateSchedulerPerBatch, self).__init__(schedule, verbose)
        self.count = 0  # Global batch index (the regular batch argument refers to the batch index within the epoch)

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        super(LearningRateSchedulerPerBatch, self).on_epoch_begin(self.count, logs)

    def on_batch_end(self, batch, logs=None):
        super(LearningRateSchedulerPerBatch, self).on_epoch_end(self.count, logs)
        self.count += 1

class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """


    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn


    def on_epoch_end(self, epoch, logs={}):

        msg = "{Epoch: %i} %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg + "\n")


class save_weights(Callback):

    def __init__(self, model, path):
        self.model = model
        self.path = path

    def on_train_begin(self, logs={}):
        self.val_sdr = []

 
    def on_epoch_end(self, epoch, logs={}):
        
        '''tmp_smry = StringIO()
        self.model.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
        summary = tmp_smry.getvalue()
        summary_split = summary.split('\n')
        summary_params = summary_split[-6:]
        summary_params = '\n'.join(summary_params)
        print('\n'+summary_params)'''

        model_clone = tf.keras.models.clone_model(self.model)
        
        self.val_sdr.append(0)

        for layer in model_clone.layers:
            layer.trainable = True

        model_clone.save_weights('/data/models/' + self.path + '/weights-' + str(int(epoch)) + '-' + str(round(logs['val_loss'], 4)) + '.hdf5')
        
        '''for layer in self.model.layers:
            if 'model_' in layer.name or 'sequential_' in layer.name:
                layer.trainable = False'''
        return
