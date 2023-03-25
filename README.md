# Requirements
pytorch>=1.2.0
# To detect single image
https://drive.google.com/file/d/15uuvIFfCdEFX1HKvL9zsByQC8aFKQD3D/view?usp=share_link

Download the weight and place it in the logs folder

run predict.py and input the image path.

We have uploaded several images in the img folder
# Dataset
www.kaggle.com/datasets/alfredzimmer/rswdatasets

Download the RSW-D dataset and place the Annotations and JPEGImages in the VOCdevkit folder

run voc_annotation.py to generate training set and test set
# Train
run train.py to train the model

the weights will be automatically saved to the logs folder
# Test
Set the weight path in yolo.py

run get_map.py
# Reference
https://github.com/bubbliiiing/yolov4-pytorch
