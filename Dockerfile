FROM python:3.7.4-alpine

MAINTAINER Loren Kuich <loren@lkuich.com>

COPY . /app
WORKDIR /app

RUN python setup.py install