# nyc-sentinel

Project for downloading and monitoring NYC satellite data from Sentinel 2 on an NVIDIA Jetson Nano.  

To start the container:

## sudo docker build -t nsk367/nyc-sentinel .

To run the container (i.e. then head to jupyter notebook and code):

## sudo docker run --runtime nvidia -it --rm --network host -v ~/nyc-sentinel:/home/noah/nyc-sentinel nsk367/nyc-sentinel