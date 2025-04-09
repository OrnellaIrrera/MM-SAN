# Use an image with GPU support and CUDA Toolkit pre-installed
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip

# Set the working directory in the container
COPY . /code/
WORKDIR /code

# Copy the requirements file into the container

RUN pip3 install pandas
RUN pip3 install nodevectors
RUN pip3 install networkx==2.6
RUN pip3 install sentence-transformers
RUN pip3 install numpy
#RUN pip3 install torch_geometric
RUN pip3 install gensim==3.6
#RUN pip3 install pybind11
#RUN pip3 install graph-walker
#RUN pip3 install --no-cache-dir -r requirements.txt
#RUN pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
#RUN pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
#RUN pip3 install torch_geometric

# Install Python dependencies


