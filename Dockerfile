FROM ubuntu:latest

# install anaconda
RUN apt-get update
RUN apt-get install -y wget && \
        apt-get clean
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH

# setup conda virtual environment
COPY requirements.yaml .
RUN conda update conda \
    && conda env create --name janus -f requirements.yaml

RUN echo "conda activate janus" >> ~/.bashrc
ENV PATH /opt/conda/envs/janus/bin:$PATH
ENV CONDA_DEFAULT_ENV $janus
