#!/usr/bin python3

import os

from flask import Flask
from flask_restful import Resource, Api, reqparse
import numpy as np

from inference import test_pred


MODEL_PATH = os.path.abspath('data/model/model.h5')

print("Loading model from: {}".format(MODEL_PATH))
clf = load(MODEL_PATH)

app = Flask(__name__)
api = Api(app)

class Prediction(Resource):
    def __init__(self):
        self._required_features = ['FILE_PATH']
        self.reqparse = reqparse.RequestParser()
        for feature in self._required_features:
            self.reqparse.add_argument(feature, 
                                                                     type=str, 
                                                                     required=True, 
                                                                     location='json',
                                                                     help = 'No {} provided'.format(feature))
        super(Prediction, self).__init__()

    def post(self):
        args = self.reqparse.parse_args()
        X = np.array([args[f] for f in self._required_features]).reshape(1, -1)
        y_pred = clf.predict(X)
        return {'prediction': y_pred.tolist()[0]}

api.add_resource(Prediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
