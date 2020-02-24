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
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from keras.layers import *
from keras import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau, EarlyStopping, ReduceLROnPlateau
from callbacks import Logger, learningratescheduler, earlystopping, reducelronplateau,LoggingCallback
import cv2
from callback import Metrics_softmask, Decoder
from helpers import text_to_labels
from aligns import Align
from models.resnet_lstm_lipread import lipreading
import numpy as np
import datetime
import pickle
import dlib
from video_utils import get_video_frames, get_frames_mouth, get_cropped_video

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

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/data/AV-speech-separation/shape_predictor_68_face_landmarks.dat')

video_file = args.video_file
transcript_file = video_file[:-3]+'align'

lips = get_cropped_video(video_file, output_dest='', detector = detector, predictor = predictor)
#lips2 = get_cropped_video('/data/grid/s2/brwg8a.mpg', output_dest='', detector = detector, predictor = predictor)
#lips3 = get_cropped_video('/data/grid/s2/brwg8a.mpg', output_dest='', detector = detector, predictor = predictor)
#lips = np.concatenate([lips, lips, lips], axis=0)
#print('Lips Concat shape:', lips.shape)
lips = crop_pad_frames(frames = lips, fps = 25, seconds = 5)
lips_lipnet = lips.reshape(1, 125,50,100,3)
print('Lipnet Input shape:', lips_lipnet.shape)

lips_resnet = []
for i in range(lips.shape[0]):
    lip_grey = cv2.cvtColor(lips[i], cv2.COLOR_RGB2GRAY)
    lips_resnet.append(lip_grey)

lips_resnet = np.asarray(lips_resnet)
lips_resnet = lips_resnet.reshape(1, 125, 50, 100, 1)
print('Resnet LSTM Input shape:', lips_resnet.shape)

# Read text

trans = np.loadtxt(transcript_file, dtype='object')
trans = trans[:, 2].tolist()
trans = ' '.join(trans)
input_length=125

lip = lipreading(mode='backendGRU', inputDim=256, hiddenDim=512, nClasses=29, frameLen=125, AbsoluteMaxStringLen=128, every_frame=True)
model_resnet = lip
model_resnet.load_weights('/data/models/combResnetLSTM_CTCloss_236k-train_1to3ratio_valWER_epochs9to20_lr1e-5_0.1decay9epochs/weights-01-110.9390.hdf5')

model_lipnet=LipNet(input_shape=(125,50,100,3), pretrained='pretrain', output_size = 29, absolute_max_string_len=128)

# Predict

pred_resnet = model_resnet.predict(lips_resnet, batch_size=1)
pred_lipnet = model_lipnet.predict(lips_lipnet, batch_size=1)

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

out_resnet = decoder.decode(pred_resnet, [125])
out_lipnet = decoder.decode(pred_lipnet, [125])

pred_resnet = np.argmax(pred_resnet, axis=2)
pred_resnet = pred_resnet.reshape(125,)
#print(pred)
letters_resnet = labels_to_text(pred_resnet.tolist())

pred_lipnet = np.argmax(pred_lipnet, axis=2)
pred_lipnet = pred_lipnet.reshape(125,)
#print(pred)
letters_lipnet = labels_to_text(pred_lipnet.tolist())

print('\n')
print('-----Resnet LSTM Predictions------')
print('Raw Output:', letters_resnet)
print('Prediction:', out_resnet[0])
print('Transcript:', trans)
print('\n')
print('--------LipNet Predictions--------')
print('Raw Output:', letters_lipnet)
print('Prediction:', out_lipnet[0])
print('Transcript:', trans)

'''import skvideo.io

outputdata = lips
writer = skvideo.io.FFmpegWriter('/data/cropped.mp4')
for i in range(outputdata.shape[0]):
    writer.writeFrame(outputdata[i, :, :, :])
writer.close()'''


