# Use uma imagem base do Ubuntu
FROM ubuntu:22.04

# Instale dependências necessárias
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    && apt-get clean
    

# Adicione o repositório do CUDA e instale o CUDA
RUN apt-get -y install nvidia-cuda-toolkit

# Instale o PyTorch
RUN apt-get install -y python3-pip
RUN pip3 install torch

# Clone o repositório desejado
RUN git clone https://github.com/brunodifranco/Time-LLM.git

# Defina o diretório de trabalho
WORKDIR /Time-LLM

# Comando padrão
CMD ["/bin/bash"]
