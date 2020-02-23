import io
import os
import uuid
from typing import Dict

import requests
import telebot
from audio_recognizer import bot, sound
from telebot import apihelper

TOKEN = os.environ["BOT_TOKEN"]
PROXY_CREDS = os.environ["BOT_PROXY"]
PROXY_SETTINGS = {
    'http': os.environ["BOT_PROXY"],
    'https': os.environ["BOT_PROXY"],
}

apihelper.proxy = PROXY_SETTINGS
bot_ = telebot.TeleBot(TOKEN)

cache: Dict[str, str] = {}  # for storing client ids


def check_client(fn):
    """Декоратор для первичной генерации uuid"""
    def wrapped(msg):
        if msg.chat.id not in cache:
            new_uuid = str(uuid.uuid4())
            cache[msg.chat.id] = new_uuid
            bot_.send_message(msg.chat.id,
                              f"Сгенерирован новый id клиента: {new_uuid}")
        fn(msg)

    return wrapped


@bot_.message_handler(content_types=["text"])
@check_client
def process_text(message):
    """Хэндлер для получения ответа от бота по тексту"""
    answer = bot.ask(message.text, uuid=cache[message.chat.id])
    bot_.reply_to(message, f"Ответ: {answer}")


@bot_.message_handler(content_types=["voice"])
@check_client
def process_sound(message):
    """Хэндлер для получения ответа от бота по голосу"""
    file_info = bot_.get_file(message.voice.file_id)
    file = requests.get(
        f'https://api.telegram.org/file/bot{TOKEN}/{file_info.file_path}',
        proxies=PROXY_SETTINGS).content
    text_repr = sound.recognize(io.BytesIO(file),
                                fmt="ogg",
                                uuid=cache[message.chat.id])
    bot_.reply_to(message, f"Вы сказали: {text_repr}")
    answer = bot.ask(text_repr, uuid=cache[message.chat.id])
    bot_.reply_to(message, f"Ответ: {answer}")


bot_.polling()

