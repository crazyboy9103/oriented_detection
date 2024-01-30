ARG TORCH_VERSION=2.1.2
ARG CUDA_VERSION=11.8
# To use nvcc, devel image must be used
FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn8-devel

RUN apt update && apt upgrade -y && apt install -y git  
WORKDIR /workspace
COPY . /workspace

RUN pip install --upgrade pip && \ 
    pip install -r requirements.txt
