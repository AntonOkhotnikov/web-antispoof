#!/usr/bin python3
import requests

url = 'http://127.0.0.1:5000/predict'
wav_path = r'data/test/test_2s.wav'

headers = {
  'content-type': 'audio/wav'
}

bytes_string = open(wav_path, 'rb')

r = requests.post(url, data=bytes_string, headers=headers)
print(r.text)
