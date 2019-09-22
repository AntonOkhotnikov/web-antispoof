#!/usr/bin python3

import os
import sys
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras import backend as K
import librosa
import soundfile as sf
import numpy as np
import random
import time
import io
from urllib.request import urlopen

sys.path.append(os.path.abspath("data/src"))
from DftSpectrogram import DftSpectrogram
from focal_loss import focal_loss
K.clear_session()


def get_feature(wav_path, length=102800, random_start=False):
    try:
        x, sr = librosa.load(wav_path, sr=16000)
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


def get_feature_from_url(wav_path, length=102800, random_start=False):
    try:
        # x, sr = librosa.load(wav_path, sr=16000)
        x, sr = sf.read(io.BytesIO(urlopen(url).read()))
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


def test_pred(model, filepath, url=False):
    if url:
        feature = get_feature_from_url(filepath, length=102800, random_start=True)
    else:
        feature = get_feature(filepath, length=102800, random_start=True)
    transfeat = feature[np.newaxis, ..., np.newaxis]
    return model.predict(transfeat)


def run_tests(items=['data/test/test_2s.wav']):
    for item in items:
        print("Making prediction on file {name}".format(name=item))
        start = time.time()
        output = test_pred(model, item)
        end = time.time()
        print('Human: {o[0][0]:.3f}\nAttack: {o[0][1]:.3f}\n'.format(o=output))  # [[human_score, spoof_score]]
        print("Time elapsed {spent:.4f}\n\n".format(spent=end - start))
    pass


if __name__ == "__main__":
    modelpath = '/home/anton/contests/boosters/deploy/web/data/model/model.h5'
    custom_objects = {'DftSpectrogram': DftSpectrogram, 'focal_loss_fixed': focal_loss()}

    model = load_model(modelpath, custom_objects=custom_objects)
    model.summary()

    items = ['data/test/test_2s.wav', 'data/test/test_25s.wav', 'data/test/test_4s.wav', 'data/test/attack_19s.wav']
    run_tests(items)
