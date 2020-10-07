FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

LABEL maintainer="Yu-Lun Chiang" \
      email="chiangyulun0914@gmail.com" \
      version="v0.1.0"

ENV PYTHONIOENCODING UTF-8

# install python 3.6 and pip
RUN apt-get update \
    && apt-get install -y python3-dev \
    && apt-get install -y python3-pip \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip

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

# RUN mkdir -p /labrador
# COPY ./src /labrador
# WORKDIR /labrador
WORKDIR /workspace

# EXPOSE 8080

# CMD /labrador/run.sh
