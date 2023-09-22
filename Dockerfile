ARG CUDA_VERSION=11.7.1
ARG UBUNTU_VERSION=20.04
# To use nvcc, devel image must be used
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# For opencv 
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 

RUN apt-get -y purge python3.8 python3-pip && apt-get -y autoremove

# Install python3.10
RUN apt-get -y install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.10 python3-pip python3.10-distutils python3.10-dev
RUN apt-get -y install build-essential pkg-config libcairo2-dev

# Symlink python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --config python3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \ 
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Install pip
RUN apt-get -y install wget
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python get-pip.py \
    && rm get-pip.py

WORKDIR /workspace
COPY . /workspace

RUN pip install --upgrade pip && \ 
    pip install -r requirements.txt
RUN pip install -e .

# RUN python --version && \
#     pip --version && \
#     pip list && \
#     python -c "import torch; print(torch.__version__); import pytorch_lightning as pl; from mmrotate._C import nms_rotated, box_iou_rotated; print(pl.__version__)"
