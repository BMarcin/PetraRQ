FROM ubuntu:20.04

# TIMEZONE
ENV TZ 'Europe/Warsaw'
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

## ENCODING TO UTF8
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV LANGUAGE C.UTF-8

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget \
    git \
    gcc \
    make \
    g++ \
    curl \
&& rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN mkdir /root/.conda
RUN chmod +x Miniconda3-latest-Linux-x86_64.sh
RUN ./Miniconda3-latest-Linux-x86_64.sh -b
RUN conda init bash
RUN rm -f Miniconda3-latest-Linux-x86_64.sh
RUN conda update -n base -c defaults conda
RUN conda install -c conda-forge -y git-annex
RUN conda create -n petrarq python=3.8
SHELL ["conda", "run", "-n", "petrarq", "/bin/bash", "-c"]
ENV PATH /root/.conda/envs/petrarq/bin:$PATH

RUN pip install unidecode --upgrade
RUN python -m pip install --upgrade pip setuptools wheel

RUN conda install -c conda-forge -y git-annex

RUN wget https://dvc.org/download/linux-deb/dvc-2.7.4
RUN dpkg -i dvc-2.7.4

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY . .

RUN chmod -R 777 /app/scripts/
RUN mkdir -p ./.dvc/tmp

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "petrarq", "/bin/bash", "-c"]
