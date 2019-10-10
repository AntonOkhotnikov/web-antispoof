#!/usr/bin python3

import os
import sys

from flask import Flask, request
from flask_restful import Resource, Api, reqparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras import backend as K

sys.path.append(os.path.abspath("data/src"))
from DftSpectrogram import DftSpectrogram
from focal_loss import focal_loss
K.clear_session()

from inference import test_pred
from bot import normalize_outputs

config = tf.ConfigProto(
    device_count={'GPU': 0},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

session = tf.Session(config=config)
K.set_session(session)


MODEL_PATH = os.path.abspath('data/model/model.h5')
custom_objects = {'DftSpectrogram': DftSpectrogram, 'focal_loss_fixed': focal_loss()}
print("Loading model from: {}".format(MODEL_PATH))
model = load_model(MODEL_PATH, custom_objects=custom_objects)
model._make_predict_function()
model.summary()


app = Flask(__name__)
api = Api(app)


class Prediction(Resource):
    def __init__(self):
        super(Prediction, self).__init__()

    def post(self):
        global session, model
        K.clear_session()
        path_to_file = request.get_data()
        print(path_to_file)
        with session.as_default():
            with session.graph.as_default():
                y_pred = test_pred(model, path_to_file, url=False, from_bytes=True)
        human, non_human = normalize_outputs(y_pred[0][0], y_pred[0][1])
        response = {'Human': '{human:.3f}'.format(human=human),
                    'Non-human': '{non_human:.3f}'.format(non_human=non_human)}
        return response


api.add_resource(Prediction, '/predict')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # port 5000 is default
