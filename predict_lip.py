import glob
import os
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

import math

import tensorflow as tf
from keras.layers import *
from keras import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau, EarlyStopping, ReduceLROnPlateau
from callbacks import Logger, learningratescheduler, earlystopping, reducelronplateau,LoggingCallback
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2
from callback import Metrics_softmask, Decoder
from generator import DataGenerator_train_softmask, DataGenerator_sampling_softmask, crop_pad_frames
from helpers import text_to_labels
from aligns import Align
#from LipNet.lipnet.model2 import LipNet
from models.resnet_lstm_lipread import lipreading
import numpy as np
import datetime
import pickle
from video_utils import get_video_frames

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

from keras.utils import multi_gpu_model
#from metrics import sdr_metric, Metrics_softmask
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('-video', action="store", dest="video_file")
#parser.add_argument('-batch_size', action="store", dest="batch_size", type=int)
#parser.add_argument('-lr', action="store", dest="lrate", type=float)

args = parser.parse_args()

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

with open("/data/AV-speech-separation/folder_filter_1.txt", "rb") as fp:  
       folders_list = pickle.load(fp) 

#folders_list_train=folders_list[0:192000]
#folders_list_val=folders_list[192000:204000]
#import random
#random.seed(10)
#random.shuffle(folders_list_train)
#folders_list_val = folders_list[91500:93000] + folders_list[238089:]
#folders_list_val=folders_list[512:768]
#random.seed(20)
#folders_list_train = random.sample(folders_list_train, 180)
#folders_list_val = random.sample(folders_list_val, 100)

#print('Training data:', len(folders_list_train)*2)
#print('Validation data:', len(folders_list_val)*2)

video_file = args.video_file
transcript_file = video_file[:-9]+'.txt'
lips = get_video_frames(video_file, fmt='rgb')
lips = crop_pad_frames(frames = lips, fps = 25, seconds = 5)
lips = lips.reshape(1, 125,50,100,3)
print('lips shape:', lips.shape)

# Read text

trans=(Align(128, text_to_labels).from_file(transcript_file))
y_data=(trans.padded_label)
y_data = y_data.reshape(1, 128)
print('y_data shape:',y_data.shape)
label_length=(trans.label_length)
input_length=125

#lip = lipreading(mode='backendGRU', inputDim=256, hiddenDim=512, nClasses=29, frameLen=125, AbsoluteMaxStringLen=128, every_frame=True)
#model = lip
model=LipNet(input_shape=(125,50,100,3), pretrained='pretrain', output_size = 29, absolute_max_string_len=128)
#model.load_weights('/data/models/combResnetLSTM_CTCloss_236k-train_1to3ratio_valWER_epochs20_lr1e-4_0.1decay9epochs/weights-07-117.3701.hdf5')

from io import StringIO
tmp_smry = StringIO()
model.summary(print_fn=lambda x: tmp_smry.write(x + '\n'))
summary = tmp_smry.getvalue()
summary_split = summary.split('\n')
summary_params = summary_split[-6:]
summary_params = '\n'.join(summary_params)
print('\n'+summary_params)

# Compile the model
#lrate = args.lrate

#adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#model = multi_gpu_model(lip.model, gpus=2)
#model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

#batch_size = args.batch_size
#epochs = args.epochs


#spell = Spell(path=PREDICT_DICTIONARY)
#decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
#                  postprocessors=[labels_to_text, spell.sentence])

#metrics_error_rates  = Statistics(lip,DataGenerator_train_softmask(folders_list_val, batch_size) , decoder, 256, output_dir='./results'))

# callcack
#metrics_wer = Metrics_softmask(model = lip, val_folders = folders_list_val, batch_size = batch_size, save_path = '/data/results/'+path+'/logs.txt')

# Fit Generator

pred = model.predict(lips, batch_size=1)

def labels_to_text(labels):
    # 26 is space, 27 is CTC blank char
    text = ''
    for c in labels:
        c1=int(c)
        if c1 >= 0 and c1 < 26:
            text += chr(c1 + ord('a'))
        elif c1 == 26:
            text += ' '
    return text

def _decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    """Decodes the output of a softmax.
    Can use either greedy search (also known as best path)
    or a constrained dictionary search.
    # Arguments
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, )` containing the sequence length for
            each batch item in `y_pred`.
        greedy: perform much faster best-path search if `true`.
            This does not use a dictionary.
        beam_width: if `greedy` is `false`: a beam search decoder will be used
            with a beam of this width.
        top_paths: if `greedy` is `false`,
            how many of the most probable paths will be returned.
    # Returns
        Tuple:
            List: if `greedy` is `true`, returns a list of one element that
                contains the decoded sequence.
                If `false`, returns the `top_paths` most probable
                decoded sequences.
                Important: blank labels are returned as `-1`.
            Tensor `(top_paths, )` that contains
                the log probability of each decoded sequence.
    """
    decoded = K.ctc_decode(y_pred=y_pred, input_length=input_length,
                           greedy=greedy, beam_width=beam_width, top_paths=top_paths)
    paths = [path.eval(session=K.get_session()) for path in decoded[0]]
    logprobs  = decoded[1].eval(session=K.get_session())

    return (paths, logprobs)

def decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1, **kwargs):
    language_model = kwargs.get('language_model', None)

    paths, logprobs = _decode(y_pred=y_pred, input_length=input_length,
                              greedy=greedy, beam_width=beam_width, top_paths=top_paths)
    if language_model is not None:
        # TODO: compute using language model
        raise NotImplementedError("Language model search is not implemented yet")
    else:
        # simply output highest probability sequence
        # paths has been sorted from the start
        result = paths[0]
    return result

class Decoder(object):
    def __init__(self, greedy=True, beam_width=100, top_paths=1, **kwargs):
        self.greedy         = greedy
        self.beam_width     = beam_width
        self.top_paths      = top_paths
        self.language_model = kwargs.get('language_model', None)
        self.postprocessors = kwargs.get('postprocessors', [])

    def decode(self, y_pred, input_length):
        decoded = decode(y_pred, input_length, greedy=self.greedy, beam_width=self.beam_width,
                         top_paths=self.top_paths, language_model=self.language_model)
        preprocessed = []
        for output in decoded:
            out = output
            for postprocessor in self.postprocessors:
                out = postprocessor(out)
            preprocessed.append(out)

        return preprocessed

PREDICT_GREEDY      = True
PREDICT_BEAM_WIDTH  = 200



decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                  postprocessors=[labels_to_text])

out = decoder.decode(pred, [125])


pred = np.argmax(pred, axis=2)
pred = pred.reshape(125,)
#print(pred)
letters = labels_to_text(pred.tolist())

#out = ''.join(letters)
print('Raw Output:', letters)
print('Prediction:', out)
print('Transcript:', trans.sentence)

#decode_res=decoder.decode(pred, 125)




