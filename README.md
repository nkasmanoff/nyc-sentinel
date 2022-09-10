# nyc-sentinel

Project for downloading and monitoring NYC satellite data from Sentinel 2 on an NVIDIA Jetson Nano. Using a ResNet-18 that can distinguish between 10 different kinds of land cover, characterize the change between the NYC region over time.


To start the container:

## sudo docker build -t nsk367/nyc-sentinel .

To run the container (i.e. then head to jupyter notebook and code):

## sudo docker run --runtime nvidia -it --rm --network host -v ~/Projects/nyc-sentinel:/home/noah/nyc-sentinel nsk367/nyc-sentinel

Once inside the container, the ML model can be trained via

## python src/trainer.py --gpus=1

To download Sentinel Images, please refer to ''Sentinel Data Download Walkthrough.ipynb''

To analyze the different regions using the trained model, please refer to ''Land Pseudo Segmentation.ipynb''