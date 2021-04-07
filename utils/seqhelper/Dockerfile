FROM ubuntu:20.04

LABEL maintainer="Yu-Lun Chiang" \
      email="chiangyulun0914@gmail.com" \
      version="v0.1.0"

ENV PYTHONIOENCODING UTF-8

# install python3.8 and pip
RUN apt-get update \
    && apt-get install -y python3.8-dev \
    && apt-get install -y python3-pip \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3.8 python \
    && pip3 install --upgrade pip

RUN apt-get install make

# Install requirements and prerequisite
RUN apt-get install -y --assume-yes --no-install-recommends \
    build-essential &&\
    rm -rf /var/lib/apt/lists/*

# copy require
COPY ./requirements.txt ./

# Install required packages
RUN pip install \
    --no-cache-dir \
    --requirement ./requirements.txt

WORKDIR /workspace

