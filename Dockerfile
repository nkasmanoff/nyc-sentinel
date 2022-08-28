FROM nvcr.io/nvidia/dli/dli-nano-ai:v2.0.1-r32.6.1


ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash


# alias python3 -> python
RUN rm /usr/bin/python && \
ln -s /usr/bin/python3 /usr/bin/python && \
ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install --upgrade pip

# Install GDAL dependencies
RUN apt-get update &&\
    apt-get install -y binutils libproj-dev gdal-bin

RUN apt-get install -y libgdal-dev

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal




WORKDIR /home/noah/nyc-sentinel
COPY requirements.txt /home/noah/nyc-sentinel/

RUN pip install -r requirements.txt
