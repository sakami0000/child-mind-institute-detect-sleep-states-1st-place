FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes software-properties-common \
    && add-apt-repository ppa:neovim-ppa/stable \
    && apt-get update \
    && apt-get install --yes  --no-install-recommends \
    build-essential \
    make \
    curl \
    cmake \
    git \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncursesw5-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    libgl1-mesa-dev \
    libopencv-dev \
    libtesseract-dev \
    libleptonica-dev \
    tesseract-ocr \
    tesseract-ocr-jpn \
    tesseract-ocr-script-jpan \
    tesseract-ocr-script-jpan-vert \
    tk-dev \
    xz-utils \
    wget \
    zip \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# python
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv \
    && ~/.pyenv/plugins/python-build/bin/python-build 3.10.11 /usr/local/python3.10.11 \
    && rm -r ~/.pyenv
ENV PATH=/usr/local/python3.10.11/bin:$PATH

# Rye
ENV RYE_HOME="/opt/rye"
ENV PATH="$RYE_HOME/shims:$PATH"
RUN curl -sSf https://rye-up.com/get | RYE_NO_AUTO_INSTALL=1 RYE_INSTALL_OPTION="--yes" bash

WORKDIR /root/app
COPY README.md pyproject.toml requirements-dev.lock requirements.lock /root/app/

RUN rye sync --no-lock

ENTRYPOINT ["rye" , "run"]
