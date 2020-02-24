import sys
sys.path.append('/data/AV-speech-separation/LipNet')
sys.path.append('/data/AV-speech-separation/models')
sys.path.append('/data/AV-speech-separation1/models/classification_models-master/')

import tensorflow as tf
import keras
from keras.layers import *
from keras import Model, Sequential
import keras.backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers.core import Lambda
#from classification_models.keras import Classifiers as Separable_Classifiers
from sep_classification_models.keras import Classifiers as Separable_Classifiers
from lipnet.core.layers import CTC
from .mish import Mish

def GRU(x, input_size, hidden_size, num_layers, num_classes, every_frame=True):

    out = Bidirectional(keras.layers.GRU(hidden_size, return_sequences=True, kernel_initializer='Orthogonal', reset_after=False, name='gru1'), merge_mode='concat')(x)
    out = Bidirectional(keras.layers.GRU(hidden_size, return_sequences=True, kernel_initializer='Orthogonal', reset_after=False, name='gru2'), merge_mode='concat')(out)
    if every_frame:
        out = Dense(num_classes, activation='softmax', name='softmax_out')(out)  # predictions based on every time step
    else:
        out = Dense(num_classes, activation='softmax', name='softmax_out')(out[:, -1, :])  # predictions based on last time-step
    return out

class Lipreading(object):
    def __init__(self, mode, inputDim=256, hiddenDim=512, nClasses=500, frameLen=28, absolute_max_string_len=128, every_frame=True):
        
        self.mode=mode 
        self.inputDim=inputDim
        self.hiddenDim=hiddenDim 
        self.nClasses=nClasses 
        self.frameLen=frameLen 
        self.absolute_max_string_len=absolute_max_string_len 
        self.every_frame=every_frame

        self.frontend3D = Sequential([
                    ZeroPadding3D(padding=(2, 3, 3)),
                    Conv3D(64, kernel_size=(5, 7, 7), strides=(1, 2, 2), padding='valid', use_bias=False, name='conv3d'),
                    BatchNormalization(),
                    ReLU(),
                    ZeroPadding3D(padding=((0, 4, 8))),
                    MaxPooling3D(pool_size=(1, 2, 3), strides=(1, 1, 2))
                    ])

        self.backend_conv1 = Sequential([
                    Conv1D(2*inputDim, 5, strides=2, use_bias=False),
                    BatchNormalization(),
                    ReLU(),
                    MaxPooling1D(2, 2),
                    Conv1D(4*inputDim, 5, strides=2, use_bias=False),
                    BatchNormalization(),
                    ReLU(),
                    ])

        self.backend_conv2 = Sequential([
                    Dense(inputDim),
                    BatchNormalization(),
                    ReLU(),
                    Dense(nClasses)
                    ])
        
        self.frames_to_batch = Lambda(lambda x : tf.reshape(x, [-1, x.shape[2], x.shape[3], x.shape[4]]), name='lambda2')

        self.nLayers=2
        self.build()
        
    def build(self):

        # Forward pass

        self.input_frames = Input(shape=(self.frameLen,50,100,1), name='frames_input')
        self.x = self.frontend3D(self.input_frames)
        print('3D Conv Out:', self.x.shape)
        self.x = self.frames_to_batch(self.x)   #x.view(-1, 64, x.size(3), x.size(4))
        print('3D Conv Out Reshape:', self.x.shape)

        self.channels = int(self.x.shape[-1])
        self.ResNet18, self.preprocess_input = Separable_Classifiers.get('resnet18')
        self.resnet18 = self.ResNet18((None, None, self.channels), weights=None, include_top=False)

        print('Resnet params:', self.resnet18.count_params())

        self.x = self.resnet18(self.x)
        print('Resnet18 Out:', self.x.shape)

        self.x = GlobalAveragePooling2D(name='global_avgpool_resnet')(self.x)
        self.x = Dense(self.inputDim)(self.x)
        self.x = BatchNormalization()(self.x)
        print('Resnet18 Linear Out:', self.x.shape)

        if self.mode == 'temporalConv':
            self.x = Lambda(lambda x : tf.reshape(x, [-1, self.frameLen, self.inputDim]), name='lambda3')(self.x)   #x.view(-1, frameLen, inputDim)

            self.x = Lambda(lambda x : tf.transpose(x, [0, 2, 1]), name='lambda4')(self.x)   #x.transpose(1, 2)
            self.x = backend_conv1(self.x)
            self.x = Lambda(lambda x : tf.reduce_mean(x, 2), name='lambda5')(self.x)
            self.x = backend_conv2(self.x)
            #print(self.x.shape)
        elif self.mode == 'backendGRU' or mode == 'finetuneGRU':
            self.x = Lambda(lambda x : tf.reshape(x, [-1, self.frameLen, self.inputDim]), name='lambda6')(self.x)    #x.view(-1, frameLen, inputDim)
            print('Input to GRU:', self.x.shape)
            self.y_pred = GRU(self.x, self.inputDim, self.hiddenDim, self.nLayers, self.nClasses, self.every_frame)
            print('GRU Out:', self.y_pred.shape)

            self.labels = Input(name='the_labels', shape=[self.absolute_max_string_len], dtype='float32')
            self.input_length = Input(name='input_length', shape=[1], dtype='int64')
            self.label_length = Input(name='label_length', shape=[1], dtype='int64')

            self.loss_out = CTC('ctc', [self.y_pred, self.labels, self.input_length, self.label_length])

        else:
            raise Exception('No model is selected')

        self.model = Model(inputs=[self.input_frames, self.labels, self.input_length, self.label_length], outputs=self.loss_out)

        return self.model
    
    def summary(self):
        Model(inputs=self.input_frames, outputs=self.y_pred).summary()
        
    def predict(self, input_batch):
        return self.test_function([input_batch, 0])[0]  # the first 0 indicates test

    @property
    def test_function(self):
        # captures output of softmax so we can decode the output during visualization
        return K.function([self.input_frames, K.learning_phase()], [self.y_pred])

def lipreading(mode, inputDim=256, hiddenDim=512, nClasses=500, frameLen=125, AbsoluteMaxStringLen=256, every_frame=True):
    model = Lipreading(mode, inputDim=inputDim, hiddenDim=hiddenDim, nClasses=nClasses, frameLen=frameLen, absolute_max_string_len=AbsoluteMaxStringLen, every_frame=every_frame)
    return model
