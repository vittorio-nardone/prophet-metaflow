FROM python:3.6.10-slim

RUN apt-get -y update  && apt-get install -y \
  python3-dev \
  apt-utils \
  python-dev \
  build-essential \
&& rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade setuptools
RUN pip install cython
RUN pip install numpy
RUN pip install pandas
RUN pip install requests
RUN pip install --upgrade plotly
RUN pip install matplotlib
RUN pip install pystan
RUN pip install fbprophet
RUN pip install metaflow

ENV USERNAME=prophet
