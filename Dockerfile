FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
#FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive



RUN apt-get update && apt-get install -y vim python3 python3-pip 

RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=40.3.0 

RUN pip3 install -U scipy scikit-learn
#RUN pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
#RUN pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install torchsummary
RUN pip3 install tensorboard==2.11.0
RUN pip3 install einops
RUN pip3 install easydict
RUN pip3 install six
RUN pip3 install timm
RUN pip3 install transformers
RUN pip3 install deepspeed==0.17.1
RUN pip3 install flash_attn==2.7.4.post1
RUN pip3 install triton==3.2.0
RUN pip3 install accelerate==1.7.0
RUN pip3 install torchcodec==0.2
RUN pip3 install peft==0.17.1

