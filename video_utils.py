import os
from os.path import join
from glob import glob 
import random
import shutil
import numpy as np
from pydub import AudioSegment
import tensorflow as tf
from scipy.io import wavfile
from scipy import signal
import math
from PIL import Image
import dlib
import skvideo.io
import time
import glob
import subprocess
import random
from PIL import Image
from scipy import signal
from scipy.io import wavfile
import math
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import shutil
home = str(Path.home())
# Avoid printing TF log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/data/AV-speech-separation/shape_predictor_68_face_landmarks.dat')

def get_frames_mouth(detector, predictor, frames):
        MOUTH_WIDTH = 100
        MOUTH_HEIGHT = 50
        HORIZONTAL_PAD = 0.19
        normalize_ratio = None
        mouth_frames = []
        for frame in frames:
            dets = detector(frame, 1)
            shape = None
            for k, d in enumerate(dets):
                shape = predictor(frame, d)
                i = -1
            if shape is None: # Detector doesn't detect face, just return as is
                return frames
            mouth_points = []
            for part in shape.parts():
                i += 1
                if i < 48: # Only take mouth region
                    continue
                mouth_points.append((part.x,part.y))
            np_mouth_points = np.array(mouth_points)

            mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

            if normalize_ratio is None:
                mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
                mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

                normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

            new_img_shape = (int(frame.shape[1] * normalize_ratio), int(frame.shape[0]* normalize_ratio))
            #resized_img = imresize(frame, new_img_shape)
            resized_img=np.array(Image.fromarray(frame).resize(new_img_shape))

            mouth_centroid_norm = mouth_centroid * normalize_ratio

            mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
            mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
            mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
            mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

            mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]

            mouth_frames.append(mouth_crop_image)
        return mouth_frames

'''def get_video_frames(path):
        videogen = skvideo.io.vreader(path)
        frames = np.array([frame for frame in videogen])
        return frames
'''
    
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


def get_cropped_video_(input_vid, output_dest, detector = detector, predictor = predictor):
    
    frames=get_video_frames(input_vid)
    shape_input = frames.shape    
    mouth=get_frames_mouth(detector, predictor, frames)
    '''s = True
    if len(mouth) == 133:
        print(mouth[-12].shape)
    for i in mouth:
        if not len(i.shape) == 3 and i.shape[0]==50 and i.shape[1] == 100 and i.shape[2] == 3:
            s = False
    print(s)'''

    try:
        
        outputdata = np.asarray(mouth)
        shape_output = outputdata.shape
    
        if shape_input != shape_output and len(shape_output) == 4:
            writer = skvideo.io.FFmpegWriter(output_dest)
            for i in range(outputdata.shape[0]):
                    writer.writeFrame(outputdata[i, :, :, :])
            writer.close()
        
    except ValueError:
        pass

def get_cropped_video(input_vid, output_dest, detector = detector, predictor = predictor):
    
    frames=get_video_frames(input_vid)
    shape_input = frames.shape    
    mouth=get_frames_mouth(detector, predictor, frames)

    try:
        
        outputdata = np.asarray(mouth)
        shape_output = outputdata.shape
    
        if shape_input != shape_output and len(shape_output) == 4:
            return outputdata
        
    except ValueError:
        pass
