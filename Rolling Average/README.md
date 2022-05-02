# Video Classification using Rolling Average

This Video Classification model is trained on top of ResNet50 backbone. Each video file is then read by using OpenCV to extract the frames at n-th interval. Each frame is centre cropped and resized to (224,224). The processed frames are then appended to form the training data frames. The model is trained using only 5 epochs and the training result is shown in the following chart.

![image](https://user-images.githubusercontent.com/6497242/139039227-97fb1d40-6f9e-4989-90db-5dcf77b6ac53.png)

The model is able to achieve >95% classification accuracy, but the drawback is the huge size of the .h5 file due to the ResNet50 architecture.

The data set used for training is from the top 5 classes from the UCF101 dataset obtained from [here](https://www.crcv.ucf.edu/data/UCF101.php)

For inference, the algorithm works in this manner:
1. Loop through the n-th frames in the video file
2. For each extracted frame, pass the frame through ResNet50 to get the inference
3. Maintain a list of the last K predictions
4. Compute the average of the last K predictions and choose the label with the largest corresponding probability
5. Label the frame and write the output frame to disk

The python script infer.py is used to do inference. The command line to run for inference is

```
python infer.py --model weights/video_classifier.h5  
                --label weights/labels.pickle 
                --input data/ 
                --output output 
                --size 128
```

Note that for the --output parameter, it is expected to be a folder to store all the processed output videos. The output video file names take after the input video filenames with an extension of _output. For example, if the input video has the file name "video_1.avi", the processed output video is automatically "video_1_output.avi" and stored in the folder defined by --output. 

Each input video is analysed for either one of the 5 classes and the class name is appended at the top of the output video.

![image](https://user-images.githubusercontent.com/6497242/139041505-a84bc584-f481-4495-8e43-bfc6ed7bace3.png)

The project is done with reference from [Video classification with Keras and Deep Learning](https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/) by Adrian Rosebrock 
