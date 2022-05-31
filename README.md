# Video Classification

Two approaches to train a classifier model to classify video categories. Data used here are from UCF101 datasets.

# 1. Video Classification using Rolling Average

This Video Classification model is trained on top of ResNet50 backbone. Each video file is then read by using OpenCV to extract the frames at n-th interval. Each frame is centre cropped and resized to (224,224). The processed frames are then appended to form the training data frames. The model is trained using only 5 epochs and the training result is shown in the following chart.

![image](https://user-images.githubusercontent.com/6497242/171235185-da5ef6f4-b453-4b51-a1d3-c37fd06fbda5.png)

The model is able to achieve >95% classification accuracy at 5 epochs.

# 2. Video Classification using RNN-CNN network

The Video Classification model is trained using both Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) consisting of GRU layers. The hybrid CNN-RNN architecture is where the CNN defines the spatial processing while the RNN defines the temporal processing. Specifically, we use the GRU layers for the Recurrent Neural Network (RNN).
