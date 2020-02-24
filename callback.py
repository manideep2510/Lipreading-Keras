import sys
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, ReduceLROnPlateau, EarlyStopping, ReduceLROnPlateau
from mir_eval.separation import bss_eval_sources
#from jiwer import wer as word_error_rate
import numpy as np
import doctest
import tensorflow as tf
from tensorflow.keras import backend as K
import glob
import random
import cv2
from aligns import Align
from helpers import text_to_labels

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

import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def get_video_frames(path, fmt='rgb'):

    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if fmt == 'rgb':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif fmt == 'grey':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = frame.reshape(frame.shape[0], frame.shape[1], 1)
            frames.append(frame)

        # Break the loop
        else: 
            break

    cap.release()
    return np.asarray(frames)

def crop_pad_frames(frames, fps, seconds):

    req_frames = fps*seconds

    num_frames = frames.shape[0]

    # Delete or add frames to make the video to 10 seconds
    if num_frames > req_frames:
        frames = frames[:req_frames, :, :, :]

    elif num_frames < req_frames:
        pad_len = req_frames - num_frames
        frames = np.pad(frames, ((0,pad_len),(0,0), (0,0), (0,0)), 'constant')

    elif num_frames == req_frames:
        frames = frames

    return frames

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
    paths = [path.numpy() for path in decoded[0]]
    logprobs = decoded[1].numpy()

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



def wer(r, h):
    """
    Source: https://martin-thoma.com/word-error-rate-calculation/

    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

def wer_sentence(r, h):
    return wer(r.split(), h.split())

def split(data):
    return data[0],data[1],data[2],data[3]



class Metrics_softmask(Callback):

    def __init__(self, model, val_folders, batch_size, save_path):
        self.model_container = model
        self.val_folders = val_folders
        self.batch_size = batch_size
        self.save_path = save_path
    def on_train_begin(self, logs={}):
        self.val_wer = []
        #self.val_f1s_weigh = []
        #self.val_recalls = []
        #self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        num = len(self.val_folders)
        div_num = 12
        num_100s = int(num/div_num)

        total_list=[]
        total_norm_list=[]
        total_wer=[]
	
        for n in range(num_100s):
            val_folders_100 = self.val_folders[n*div_num:(n+1)*div_num]
            lips=[]
            transcripts=[]
            for folder in val_folders_100:

                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)

                transcripts_ = sorted(glob.glob(folder + '/*.txt'), key=numericalSort)

                lips.append(lips_[0])
                lips.append(lips_[1])

                transcripts.append(transcripts_[0])
                transcripts.append(transcripts_[1])

            zipped = list(zip(lips, transcripts))
            random.shuffle(zipped)
            lips, transcripts = zip(*zipped)



            X_lips = []

            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt='grey')
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 5)
                X_lips.append(x_lips)

            align=[]
            Y_data = []
            label_length = []
            input_length = []
            source_str = []
            X_lips = np.asarray(X_lips)


            for i in range(len(transcripts)):align.append(Align(128, text_to_labels).from_file(transcripts[i]))
            for i in range(X_lips.shape[0]):
                Y_data.append(align[i].padded_label)
                label_length.append(align[i].label_length)
                input_length.append(X_lips.shape[1])
                #source_str.append(align[i].sentence)
            Y_data = np.array(Y_data)

            val_predict=self.model_container.predict(X_lips)

        #
        # for n in range(num_100s):
        #     val_folders_100 = self.val_folders[n*100:(n+1)*100]
        #     d0,d1,d2,d3=split(DataGenerator_test(val_folders_100, self.batch_size))
        #     val_predict = (self.model.predict(d0))

            decode_res=decoder.decode(val_predict, input_length)

            ground_truth=[]
            for i in range(Y_data.shape[0]):
                ground_truth.append(labels_to_text(Y_data[i]))

            data=[]
            for j in range(0, X_lips.shape[0]):
                data.append((decode_res[j], ground_truth[j]))


            mean_individual_length = np.mean([len(pair[1].split()) for pair in data])
            total       = 0.0
            total_norm  = 0.0
            w=0.0
            length      = len(data)
            for i in range(0, length):
                val         = float(wer_sentence(data[i][0], data[i][1]))
                total      += val
                total_norm += val / mean_individual_length
                w+=val/len(data[i][1])

            total_wer.append(w/length)
            total_list.append(total/length)
            total_norm_list.append(total_norm/length)
        total_wer=np.array(total_wer)
        total_list=np.array(total_list)
        total_norm_list=np.array(total_norm_list)

        print('Validation WER_original:',np.mean(total_wer),'Validation WER: ', np.mean(total_list),'Validation WER_NORM:',np.mean(total_norm_list))
        
        with open(self.save_path, "a") as myfile:
            myfile.write(', Validation WER_original: ' + str(np.mean(total_wer)) + ', Validation WER: ' + str(np.mean(total_list)) + ', Validation WER_NORM: ' + str(np.mean(total_norm_list)) + '\n')

#             return self.get_mean_tuples(data, mean_individual_length, wer_sentence)
#
#             def get_mean_tuples(self, data, individual_length, func):
#                 total       = 0.0
#                 total_norm  = 0.0
#                 length      = len(data)
#                 for i in range(0, length):
#                     val         = float(func(data[i][0], data[i][1]))
#                     total      += val
#                     total_norm += val / individual_length
#                 return (total/length, total_norm/length)
#
#
#             mixed_spect = val_predict[:,:,:,1]
#             mixed_phase = val_predict[:,:,:,2]
#             val_targ = val_predict[:,:,:,3]
#             batch = val_targ.shape[0]
#             val_targ = val_targ.reshape(batch, -1)
# #           val_targ = val_targ[:, :80000]
#
#             masks = val_predict[:,:,:,0]
#
#             samples_pred = []
#             for i in range(masks.shape[0]):
#                 mask = masks[i]
#                 #print('mask', mask.shape)
#                 mixed_spect_ = mixed_spect[i]
#                 #print('mixed_spect_' ,mixed_spect_.shape)
#                 mixed_phase_ = mixed_phase[i]
#                 #print('mixed_phase_', mixed_phase_.shape)
#                 samples = retrieve_samples(spec_signal = mixed_spect_,phase_spect = mixed_phase_,mask = mask,sample_rate=16e3, n_fft=512, window_size=25, step_size=10)
#
#                 #print('samples', samples.shape)
#                 samples_pred.append(samples[256:])
#
#             val_targ1 = []
#             for i in range(batch):
#                 length_pred = len(samples_pred[i])
#                 #print('length_pred', length_pred)
#                 val_targ_ = val_targ[i, :length_pred]
#                 #val_targ_ = val_targ_.reshape(1, -1)
#                 #print('val_targ_', val_targ_.shape)
#                 val_targ1.append(val_targ_)
#
#             val_targ = val_targ1
#
#             samples_pred = np.asarray(samples_pred)
#             #print('samples_pred', samples_pred.shape)
#             val_targ = np.asarray(val_targ)
#             #print('val_targ', val_targ.shape)
#             #val_predict = val_predict1
#             #val_targ = val_targ1
#             #_val_f1 = f1_score(val_targ, val_predict)
#             #_val_f1_weigh = f1_score(val_targ, val_predict, average='weighted')
#             #_val_recall = recall_score(val_targ, val_predict)
#             #_val_precision = precision_score(val_targ, val_predict)
#
#             _val_sdr1, _ = metric_eval(target_samples = val_targ, predicted_samples = samples_pred)
#             sdr_list.append(_val_sdr1)
#
#         sdr_list = np.asarray(sdr_list)
#         _val_sdr = np.mean(sdr_list)
#         self.val_sdr.append(_val_sdr)
#         #self.val_f1s_weigh.append(_val_f1_weigh)
#         #self.val_recalls.append(_val_recall)
#         #self.val_precisions.append(_val_precision)
# #        print '\n'
#         print('Validation SDR: ', _val_sdr)
        #print('Weighted validation f1: ', _val_f1_weigh)
        #, '_val_precision: ', _val_precision, '_val_recall', _val_recall
        return


class Metrics_lipnet(Callback):

    def __init__(self, model, val_folders, batch_size, save_path):
        self.model_container = model
        self.val_folders = val_folders
        self.batch_size = batch_size
        self.save_path = save_path
    def on_train_begin(self, logs={}):
        self.val_wer = []
        #self.val_f1s_weigh = []
        #self.val_recalls = []
        #self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        num = len(self.val_folders)
        div_num = self.batch_size
        num_100s = int(num/div_num)

        total_list=[]
        total_norm_list=[]
        total_wer=[]
	
        for n in range(num_100s):
            val_folders_100 = self.val_folders[n*div_num:(n+1)*div_num]
            lips=[]
            transcripts=[]
            for folder in val_folders_100:

                #lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)

                #transcripts_ = sorted(glob.glob(folder + '/*.txt'), key=numericalSort)

                lips.append(folder)
                #lips.append(lips_[1])

                transcripts.append(folder[:-9]+'.txt')
                #transcripts.append(transcripts_[1])

            zipped = list(zip(lips, transcripts))
            random.shuffle(zipped)
            lips, transcripts = zip(*zipped)



            X_lips = []

            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt='grey')
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 5)
                X_lips.append(x_lips)

            align=[]
            Y_data = []
            label_length = []
            input_length = []
            source_str = []
            X_lips = np.asarray(X_lips)


            for i in range(len(transcripts)):align.append(Align(128, text_to_labels).from_file(transcripts[i]))
            for i in range(X_lips.shape[0]):
                Y_data.append(align[i].padded_label)
                label_length.append(align[i].label_length)
                input_length.append(X_lips.shape[1])
                #source_str.append(align[i].sentence)
            Y_data = np.array(Y_data)

            val_predict=self.model_container.predict(X_lips)

        #
        # for n in range(num_100s):
        #     val_folders_100 = self.val_folders[n*100:(n+1)*100]
        #     d0,d1,d2,d3=split(DataGenerator_test(val_folders_100, self.batch_size))
        #     val_predict = (self.model.predict(d0))

            decode_res=decoder.decode(val_predict, input_length)

            ground_truth=[]
            for i in range(Y_data.shape[0]):
                ground_truth.append(labels_to_text(Y_data[i]))

            data=[]
            for j in range(0, X_lips.shape[0]):
                data.append((decode_res[j], ground_truth[j]))


            mean_individual_length = np.mean([len(pair[1].split()) for pair in data])
            total       = 0.0
            total_norm  = 0.0
            w=0.0
            length      = len(data)
            for i in range(0, length):
                val         = float(wer_sentence(data[i][0], data[i][1]))
                total      += val
                total_norm += val / mean_individual_length
                w+=val/len(data[i][1])

            total_wer.append(w/length)
            total_list.append(total/length)
            total_norm_list.append(total_norm/length)
            
        total_wer=np.array(total_wer)
        total_list=np.array(total_list)
        total_norm_list=np.array(total_norm_list)

        print('Validation WER_original:',np.mean(total_wer),'Validation WER: ', np.mean(total_list),'Validation WER_NORM:',np.mean(total_norm_list))
        
        with open(self.save_path, "a") as myfile:
            myfile.write(', Validation WER_original: ' + str(np.mean(total_wer)) + ', Validation WER: ' + str(np.mean(total_list)) + ', Validation WER_NORM: ' + str(np.mean(total_norm_list)) + '\n')


class Metrics_cotrain(Callback):

    def __init__(self, model, val_folders, batch_size, save_path):
        self.model_container = model
        self.val_folders = val_folders
        self.batch_size = batch_size
        self.save_path = save_path
    def on_train_begin(self, logs={}):
        self.val_wer = []
        #self.val_f1s_weigh = []
        #self.val_recalls = []
        #self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        num = len(self.val_folders)
        div_num = 12
        num_100s = int(num/div_num)

        total_list=[]
        total_norm_list=[]
        total_wer=[]
	
        for n in range(num_100s):
            val_folders_100 = self.val_folders[n*div_num:(n+1)*div_num]
            lips=[]
            transcripts=[]
            samples = []
            samples_mix = []
            for folder in val_folders_100:

                lips_ = sorted(glob.glob(folder + '/*_lips.mp4'), key=numericalSort)
                samples_ = sorted(glob.glob(folder + '/*_samples.npy'), key=numericalSort)
                samples_mix_ = '/data/mixed_audio_files/' +folder.split('/')[-1]+'.wav'
                transcripts_ = sorted(glob.glob(folder + '/*.txt'), key=numericalSort)

                '''lips.append(lips_[0])
                lips.append(lips_[1])

                transcripts.append(transcripts_[0])
                transcripts.append(transcripts_[1])'''

                for i in range(len(lips_)):
                    lips.append(lips_[i])
                for i in range(len(samples_)):
                    samples.append(samples_[i])
                for i in range(len(lips_)):
                    samples_mix.append(samples_mix_)
                for i in range(len(lips_)):
                    transcripts.append(transcripts_[i])

            zipped = list(zip(lips, samples, samples_mix, transcripts))
            random.shuffle(zipped)
            lips, samples, samples_mix, transcripts = zip(*zipped)

            X_samples = np.asarray([np.pad(np.load(fname), (0, 32000), mode='constant')[:32000] for fname in samples])
            X_samples_mix = np.asarray([np.pad(wavfile.read(fname)[1], (0, 32000), mode='constant')[:32000] for fname in samples_mix])

            X_lips = []

            for i in range(len(lips)):

                x_lips = get_video_frames(lips[i], fmt='grey')
                x_lips = crop_pad_frames(frames = x_lips, fps = 25, seconds = 2)
                X_lips.append(x_lips)

            X_lips = np.asarray(X_lips)

            align=[]
            Y_data = []
            label_length = []
            input_length = []
            source_str = []

            for i in range(len(transcripts)):align.append(Align(128, text_to_labels).from_file(transcripts[i]))
            for i in range(X_lips.shape[0]):
                Y_data.append(align[i].padded_label)
                label_length.append(align[i].label_length)
                input_length.append(X_lips.shape[1])
                #source_str.append(align[i].sentence)
            Y_data = np.array(Y_data)

            X_samples_targ = X_samples.reshape(X_samples.shape[0], 32000, 1).astype('float32')
            X_samples_mix = X_samples_mix.reshape(X_samples_mix.shape[0], 32000, 1).astype('float32')
            X_samples_targ = X_samples_targ/1350.0
            X_samples_mix = X_samples_mix/1350.0

            val_predict=self.model_container.predict([X_lips, X_samples_mix])

            val_predict = val_predict[1]

        #
        # for n in range(num_100s):
        #     val_folders_100 = self.val_folders[n*100:(n+1)*100]
        #     d0,d1,d2,d3=split(DataGenerator_test(val_folders_100, self.batch_size))
        #     val_predict = (self.model.predict(d0))

            decode_res=decoder.decode(val_predict, input_length)

            ground_truth=[]
            for i in range(Y_data.shape[0]):
                ground_truth.append(labels_to_text(Y_data[i]))

            data=[]
            for j in range(0, X_lips.shape[0]):
                data.append((decode_res[j], ground_truth[j]))


            mean_individual_length = np.mean([len(pair[1].split()) for pair in data])
            total       = 0.0
            total_norm  = 0.0
            w=0.0
            length      = len(data)
            for i in range(0, length):
                val         = float(wer_sentence(data[i][0], data[i][1]))
                total      += val
                total_norm += val / mean_individual_length
                w+=val/len(data[i][1])

            total_wer.append(w/length)
            total_list.append(total/length)
            total_norm_list.append(total_norm/length)
        total_wer=np.array(total_wer)
        total_list=np.array(total_list)
        total_norm_list=np.array(total_norm_list)

        print('Validation WER_original:',np.mean(total_wer),'Validation WER: ', np.mean(total_list),'Validation WER_NORM:',np.mean(total_norm_list))

        return
