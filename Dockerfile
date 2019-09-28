FROM continuumio/miniconda3:latest
MAINTAINER Anton Okhotnikov

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get update -qq \
    && apt-get install -y vim libsndfile1

COPY ./* /home/web-antispoof/

WORKDIR /home/web-antispoof

RUN pip install -r requirements.txt

# create directory to store temporary audio files
RUN mkdir /opt/audio

ENTRYPOINT python3 bot.py