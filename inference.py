#!/usr/bin python3

import os
import sys
from keras.models import load_model
from keras import backend as K
import librosa
import numpy as np
import random
import time

sys.path.append(os.path.abspath("data/src"))
from DftSpectrogram import DftSpectrogram
from focal_loss import focal_loss
K.clear_session()


modelpath = '/home/anton/contests/boosters/deploy/web/data/model/model.h5'
custom_objects = {'DftSpectrogram': DftSpectrogram, 'focal_loss_fixed': focal_loss()}

model = load_model(modelpath, custom_objects=custom_objects)
model.summary()


def get_feature(wav_path, length=102000, random_start=False):
    try:
        x, sr = librosa.load(wav_path, sr=None)
        assert sr == 16000
        if length > len(x):
            x = np.concatenate([x] * int(np.ceil(length/len(x))))
        if random_start:
            x = x[random.randint(0, len(x) - length):]
        feature = x[:length]
        return feature / np.max(np.abs(feature))
    except Exception as e:
        print("Error with getting feature from %s: %s" % (wav_path, str(e)))
        return None


def test_pred(model, filepath):
    print('making test prediction')
    feature = get_feature(filepath, length=102800, random_start=True)
    print(feature.shape)
    transfeat = feature[np.newaxis, ..., np.newaxis]
    print(transfeat.shape)
    return model.predict(transfeat)

start = time.time()
output = test_pred(model, 'data/test/spoof_00002.wav')
end = time.time()
print(output)  # [human_score, spoof_score]
print("Time elapsed {spent:.4f}".format(spent=end - start))