FROM python:3.7

MAINTAINER Loren Kuich <loren@lkuich.com>

COPY . /app
WORKDIR /app

RUN python setup.py install
