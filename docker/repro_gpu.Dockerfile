FROM nvidia/cuda:11.0.3-runtime-ubuntu20.04

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
    gfortran\
    libopenblas-dev\
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
RUN mkdir /root/.conda
RUN chmod +x Miniconda3-py39_4.10.3-Linux-x86_64.sh
RUN ./Miniconda3-py39_4.10.3-Linux-x86_64.sh -b
RUN conda init bash
RUN rm -f Miniconda3-py39_4.10.3-Linux-x86_64.sh
#RUN conda update -n base -c defaults conda
#RUN conda install -c conda-forge pyarrow
RUN conda install -c conda-forge pycosat
#RUN conda create -n petrarq python=3.9
RUN conda install -c conda-forge -y git-annex
RUN conda install -c conda-forge lapack
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
#ENV PATH /root/.conda/envs/petrarq/bin:$PATH

RUN pip install unidecode --upgrade
RUN python -m pip install --upgrade pip setuptools wheel

RUN wget https://dvc.org/download/linux-deb/dvc-2.7.4
RUN dpkg -i dvc-2.7.4

COPY ../requirements.txt ./requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY .. .

RUN chmod -R 777 /app/scripts/
RUN mkdir -p ./.dvc/tmp

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "base", "/bin/bash", "-c"]