## Behavioral Cloning

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./examples/nvidia_architecture.png "Model Visualization"
[image2]: ./examples/center.jpg "center of lane"
[image3]: ./examples/left.jpg "left of lane"
[image4]: ./examples/right.jpg "right of lane"
[image5]: ./examples/center_flipped.jpg "flipped center of lane"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on Nvidia archictecture with 5 convolution layers and 3 fully connected layers. 5x5 and 3x3 filter sizes and depths between 24 and 64 are used on the convoluations layers.

The model includes RELU activation function in the convolution layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 118). 

#### 2. Attempts to reduce overfitting in the model

The model contains a one dropout layer in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track as seen on video.mp4.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 134).

#### 4. Appropriate training data

Training data consist of raw driving data provided from Udacity and my own driving with 2 laps of center lane driving on track 1. I also captured data for driving situation around sharp turns. I used combined data of center lane driving, recovering from the left and right sides of the road into the final training data set.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use a simple convolution neural network model with the focus on getting basic/low prediction of steering angle base on center/middle of lane driving images and be able apply the model prediction to the autonomous driving mode in simulator to get the car moving.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I revised the model to adapt the features from the Nvidia convolution archicture.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially at sharp turns. To improve the driving behavior in these cases, I drove these problematic spots and catpured the driving data and include these data in training data set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 117-137) is a convolution neural network with the following layers and layer sizes:

    
    Layer (type)                 Output Shape             Param #   
    _________________________________________________________________
    lambda_1 (Lambda)            (None, 80, 320, 3)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 38, 158, 24)       1824      
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 17, 77, 36)        21636     
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 7, 37, 48)         43248     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 7, 37, 48)         0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 4, 19, 64)         27712     
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 1, 9, 64)          36928     
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 576)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 100)               57700     
    _________________________________________________________________
    dense_2 (Dense)              (None, 50)                5050      
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                510       
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 11        




Here is a visualization of the architecture

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

The model also utilized the images captured from the vehicle's left and right side cameras/perspective and apply a steering offset factor of 0.2 from the center steering angle (the offset factor is added to center steering angle for to left camera image and  subtracted from center steering angle for right camera image). This data adjustment help the model learn to steer the vehicle back to center of lane. These images show what a recovery looks like from left and right side cameras:


![alt text][image3]
![alt text][image4]

To further augment the data set, the same images and related angles are flipped horizontally to produce different data without making extra driving laps but the flipped data still provide new perspective/driving environment features for the model to learn. Hear are sample images that has then been flipped:

center of lane
![alt text][image2]

center of lane - flipped
![alt text][image5]


After the collection process, I preprocessed the images by cropping and normalizing.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the Mean Square Error was not much improving after 5th epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.