#!/usr/bin/python
'''
Author: SARBAJIT MUKHERJEE
Email: sarbajit.mukherjee@aggiemail.usu.edu
$python data_preparation.py
This code prepares the raw audio data and packs it as a pickle file to be
used in Tflearn
'''

import pydub
pydub.AudioSegment.ffmpeg = '/usr/bin/ffmpeg'
import pickle
import numpy as np
import os
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from scipy import interpolate

AUDIO_LENGTH = 20000
NEW_SAMPLERATE = 12000


###########Resampling raw audio to 12Khz and then returing an array#################
def extract_feature(file_name):
    try:

        old_samplerate, old_audio = wavfile.read(file_name)
        if old_samplerate != NEW_SAMPLERATE:
            duration = old_audio.shape[0] / old_samplerate

            time_old = np.linspace(0, duration, old_audio.shape[0])
            time_new = np.linspace(0, duration, int(old_audio.shape[0] * NEW_SAMPLERATE / old_samplerate))

            interpolator = interpolate.interp1d(time_old, old_audio.T)
            new_audio = interpolator(time_new).T

            wavfile.write(file_name, NEW_SAMPLERATE, np.round(new_audio).astype(old_audio.dtype))

        sample_rate, X = wavfile.read(file_name)
        X = X.reshape(-1, 1)
        audio_buf = (X - np.mean(X)) / np.std(X)
        original_length = len(audio_buf)
        if original_length < AUDIO_LENGTH:
            audio_buf = np.concatenate((audio_buf, np.zeros(shape=(AUDIO_LENGTH - original_length, 1))))
        elif original_length > AUDIO_LENGTH:
            audio_buf = audio_buf[0:AUDIO_LENGTH]
        return audio_buf
    except:
        print('wav error: ' + file_name)
        return None

i = 0
data ={}
Y = []

###########Buzz Train#################
buzz_path = 'Data/BUZZ1/bee' 

for root, dirs, files in os.walk(buzz_path):
    for item in files:
        if item.endswith('.wav'):
            x = os.path.join(root, item)
            spec_pad = extract_feature(x)
            ext_features = np.hstack([spec_pad])
            ext_features = np.array(ext_features)
            data[i] = ext_features
            Y.append(int(1))
        i = i+1
######################################

###########Noise Train#################
noise_path = 'Data/BUZZ1/noise'

for root, dirs, files in os.walk(noise_path):
    for item in files:
        if item.endswith('.wav'):
            x = os.path.join(root, item)
            spec_pad = extract_feature(x)
            ext_features = np.hstack([spec_pad])
            ext_features = np.array(ext_features)
            data[i] = ext_features
            Y.append(int(0))
        i = i+1
######################################

###########Cricket Train#################
cricket_path = 'Data/BUZZ1/cricket'

for root, dirs, files in os.walk(cricket_path):
    for item in files:
        if item.endswith('.wav'):
            x = os.path.join(root, item)
            spec_pad = extract_feature(x)
            ext_features = np.hstack([spec_pad])
            ext_features = np.array(ext_features)
            data[i] = ext_features
            Y.append(int(2))
        i = i+1
######################################

###########Data Preprocessing################

X = np.array(data.values())
X_ = [val for val in data.values()]

#X, Y = shuffle(X, Y)
#############################################

Y = F.one_hot(torch.IntTensor(Y).to(torch.int64))
pickle_path = 'pickle/'

#######writing to pickel#####################
np.savez(pickle_path + 'train_data.npz', data = X_, labels = Y)

# f = open(pickle_path+'train_labels.pickle', 'wb')
# pickle.dump(Y, f)
# f.close()

#############################################

# print (X.shape)