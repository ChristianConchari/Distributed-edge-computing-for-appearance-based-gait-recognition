# CNN-Gait-Recognition-Offline

This repository contains the code for the Distributed edge computing for appearance-based gait recognition reserach. The code is divided in two main parts: The Offline part and the Online part. The Offline part is in charge of the dataset preprocessing, the fine-tuning of the instance segmentation model and the training of the gait recognition model. The Online part is in charge of the inference of the gait recognition model and the communication with the edge devices.

<div style="text-align: center;">
    <img src="utils/090-008-bg-bg-08.gif" alt="drawing" style="width:600px;"/>
    <p><em>Figure: Gait Recognition Example</em></p>
</div>

## Repository structure

### Offline processing

Inside the Offline-processing folder you will find four notebooks:

- **OAKGait16-data-generation.ipynb**: This notebook will contain the steps for preprocessing the OAK Gait 16 dataset. The dataset is divided in two parts: The training set and the testing set. The training set is used to feed the offline framework.

- **segmentation-data-generation.ipynb**: This notebook will contain the steps for preprocessing the OAK Gait 16 dataset in order to generate the necessary data for the fine-tuning of the instance segmentation model.

- **segmentation-model-training.ipynb**: This notebook will contain the steps for fine-tuning the instance segmentation model. The model is fine-tuned using the OAK Gait 16 dataset.

- **gait-recognition-model-training.ipynb**: This notebook will contain the steps for training the gait recognition classification model. In this notebook, a CNN is trained using the OAK Gait 16 dataset in order to perform classification of the gait sequences.
 
### Online processing

Inside the Online-processing folder you will find two directories:

- **edge-node-inference**: This directory contains the code for the edge node. The edge node is in charge of the inference of the gait recognition model. Depending if the edge node will run on the test videos or on the live feed, the code will be also communicate with the edge-server.

- **edge-server-gui**: This directory contains the code for the edge server. The edge server is in charge of the communication with the edge nodes and showing the inference results in a GUI.
