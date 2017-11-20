#**Behavioral Cloning** 



---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./images/training-validation-errors.png "Training/Validation Error"
[image2]: ./images/center.jpg "Center Camera"
[image3]: ./images/left.jpg "Left Camera"
[image4]: ./images/right.jpg "Right Camera"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I used the model described in https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

After few unsuccessful iterations using LeNet first then exploring with other networks I decided to focus on the
nvidia model as it was proven to give good results. 


####2. Attempts to reduce overfitting in the model

I tried to use Dropout but it didn't give me good results so the final model does not include any dropout.
I kept the epochs low (3 epochs only) to avoid overfitting.
The reasoning behind this is that after 3 training the training rate dropped but the validation error increased 
which is a sing that past 3-4 epochs the model was starting to overfit my training examples.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to experiment with Different Network architectures until getting to a functioning architecture.

I tuned few parameters such as the correction (binary search from 0.1 to 0.2 give me 0.15 as a decent value)
and the number of epochs (3 epochs seemed to be the right amount of training that the network needed before it started to overfit my training data)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of the following


Below is the Model summary:

|Preprocessing and Normlization   |
|---------------------------------|
|Cropping upper and bottom pixels at 70 and 25 respectively        |
||
|24x5x5 Convolutional layer, Stride = 2                       |
||
|Relu Activation                  |
||
|36x5x5 Convolutional layer, Stride = 2                       |
||
|Relu Activation                  |
||
|48x5x5 Convolutional layer, Stride = 2                       |
||
|Relu Activation                  |
||
|64x3x3 Convolutional layer, Stride = 1                       |
||
|Relu Activation                  |
||
|64x3x3 Convolutional layer, Stride = 1                       |
||
|Relu Activation                  |
||
|Flatten                          |
||
|100 Neurons Fully Connceted      |
||
|50 Neurons Fully Connceted       |
||
|10 Neurons Fully Connceted       |
||
|1 Neuron for output              |
||

This is the model described in the Nvidia paper suggested in the class


####3. Creation of the Training Set & Training Process



Training Data:

I drove about two tracks and recorded it
I turned 180 degrees 
I drove two more tacks and recorded it

Below few images of my training Data:


![alt text][image2]
![alt text][image3]
![alt text][image4]



After the collection process, I had 34368 number of data points. I then preprocessed this data by normalizing it
and mean-centering it.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3.

Below is a graph illustrating this:

![alt text][image1]


I used an adam optimizer so that manually training the learning rate wasn't necessary.
