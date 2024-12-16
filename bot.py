#!/usr/bin python3

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras import backend as K

from telegram.ext import Updater
from telegram.ext import CommandHandler, MessageHandler, Filters
import logging

from inference import test_pred

sys.path.append(os.path.abspath("data/src"))
from DftSpectrogram import DftSpectrogram
from focal_loss import focal_loss
K.clear_session()


# Telegram bot token
TOKEN = '<YOUR_TELEGRAM_BOT_TOKEN_GOES_HERE>'

# proxy server
REQUEST_KWARGS = {
    'proxy_url': 'http://USERNAME:PASSWORD@YOUR_IP_ADDRESS:PORT',
}

# Path to antispoofing model
MODEL_PATH = os.path.abspath('data/model/model.h5')

# Path to store all the downloaded audio
DATA_ROOT = os.path.abspath('/opt/audio')


def initialize_model(model_path):
    config = tf.ConfigProto(
        device_count={'GPU': 0},
        intra_op_parallelism_threads=1,
        allow_soft_placement=True
    )

    session = tf.Session(config=config)
    K.set_session(session)

    custom_objects = {'DftSpectrogram': DftSpectrogram, 'focal_loss_fixed': focal_loss()}
    print("Loading model from: {}".format(MODEL_PATH))
    model = load_model(model_path, custom_objects=custom_objects)
    model._make_predict_function()
    model.summary()
    return model, session


def initialize_bot(token, request_kwargs=None):
    # if you have and require a proxy please set the request kwargs
    if request_kwargs is not None:
        updater = Updater(token=token, use_context=True, request_kwargs=request_kwargs)
    else:
        updater = Updater(token=token, use_context=True)
    dispatcher = updater.dispatcher
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)

    echo_handler = MessageHandler(Filters.text, echo)
    dispatcher.add_handler(echo_handler)

    voice_handler = MessageHandler(Filters.voice, voice)
    dispatcher.add_handler(voice_handler)

    audio_handler = MessageHandler(Filters.audio, audio)
    dispatcher.add_handler(audio_handler)

    document_handler = MessageHandler(Filters.document.category("audio"), document)
    dispatcher.add_handler(document_handler)

    return updater, dispatcher


def start(update, context):
    text = "Please send me a voice message or an audio file"
    context.bot.send_message(chat_id=update.message.chat_id, text=text)


def echo(update, context):
    text = "Please send me a voice message or an audio file"
    context.bot.send_message(chat_id=update.message.chat_id, text=text)


def download_voice(update, context):
    global DATA_ROOT
    # get current chat_id
    chat_id = update.message.chat_id
    # check if folder exists for the given chat_id
    out_path = os.path.join(DATA_ROOT, str(chat_id))
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    # make a filename
    file_name = "/".join([out_path, "voice.ogg"])
    file_id = update.message.voice.file_id
    audio_file = context.bot.get_file(file_id)
    audio_file.download(file_name)
    return file_name


def download_attachment(update, context, file_info):
    global DATA_ROOT
    # get current chat_id
    chat_id = update.message.chat_id
    # check if folder exists for the given chat_id
    out_path = os.path.join(DATA_ROOT, str(chat_id))
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    # make a filename
    ext = file_info.file_path.split(".")[-1]
    file_name = "/".join([out_path, f"audio.{ext}"])
    file_id = file_info.file_id
    audio_file = context.bot.get_file(file_id)
    audio_file.download(file_name)
    return file_name


def voice(update, context):
    context.bot.send_message(chat_id=update.message.chat_id, text="Received a voice message")
    file_info = context.bot.get_file(update.message.voice.file_id)
    print(file_info)
    file_path = download_voice(update, context)
    score_audio(update, context, file_path)


def audio(update, context):
    context.bot.send_message(chat_id=update.message.chat_id, text="Received an audio file")
    file_info = context.bot.get_file(update.message.audio.file_id)
    print(file_info)
    file_path = download_attachment(update, context, file_info)
    score_audio(update, context, file_path)


def document(update, context):
    context.bot.send_message(chat_id=update.message.chat_id, text="Received a document")
    file_info = context.bot.get_file(update.message.document.file_id)
    print(file_info)
    file_path = download_attachment(update, context, file_info)
    score_audio(update, context, file_path)


def normalize_outputs(x, y):
    return x / (x + y), y / (x + y)


def score_audio(update, context, file_path):
    global model, session
    K.clear_session()
    with session.as_default():
        with session.graph.as_default():
            y_pred = test_pred(model, file_path, url=False)
    human, non_human = normalize_outputs(y_pred[0][0], y_pred[0][1])
    response = 'Human: {human:.3f}\nNon-human: {non_human:.3f}'.format(human=human, non_human=non_human)
    context.bot.send_message(chat_id=update.message.chat_id, text=response)

# Add handling of telegram.Audio and telegram.Document and Filters.forwarded and Filters.voice
# maybe merge them into one  telegram.ext.filters.MergedFilter(base_filter, and_filter=None, or_filter=None)

# updater.start_webhook()


if __name__ == "__main__":
    model, session = initialize_model(MODEL_PATH)

    updater, dispatcher = initialize_bot(TOKEN, request_kwargs=None)  # specify proxy in REQUEST_KWARGS if required

    # start telegram bot
    updater.start_polling()
