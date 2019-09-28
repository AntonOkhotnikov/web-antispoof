FROM continuumio/miniconda3:latest
MAINTAINER Anton Okhotnikov

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get update -qq \
    && apt-get install -y vim libsndfile1 libsndfile-dev

COPY ./* /home/web-antispoof/

COPY data /home/web-antispoof/data/

WORKDIR /home/web-antispoof

RUN pip install -r requirements.txt \
    && conda install -c conda-forge libsndfile

# create directory to store temporary audio files
RUN mkdir /opt/audio

CMD python3 bot.py
