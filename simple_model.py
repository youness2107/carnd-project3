import csv
import cv2
import numpy as np

lines = []

# Loading the training set images
with open('data/set2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Populating the images and their corresponding steering angles
# These two populated arrays will represent the ground truth for the model
# which it will learn from
images = []
measurements = []
for line in lines:
    image_center = cv2.imread(line[0])
    image_left = cv2.imread(line[1])
    image_right = cv2.imread(line[2])
    steering_center = float(line[3])
    # Correction to add/subtract from the center steeting
    # when using left and right camera images
    correction = 0.15
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    images.append(image_center)
    images.append(image_left)
    images.append(image_right)
    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)
    # easy way to agument data by adding the horizontally 
    # flipped images and the opposite steering 
    images.append(cv2.flip(image_center, 1))
    images.append(cv2.flip(image_left, 1))
    images.append(cv2.flip(image_right, 1))
    measurements.append(steering_center*-1.0)
    measurements.append(steering_left*-1.0)
    measurements.append(steering_right*-1.0)
    
    
	
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D

model = Sequential()
# Simple Data preprocessing 
# normalizing the image and mean-centering it
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# remove top 70 pixels and bottom 25 pixels
model.add(Cropping2D(cropping=((70,25), (0,0))))

# Adding the model described in the nvidia paper and also
# explained in the class 
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Using MSE instead of Cross Entropy since this is not a classification task
# Using the Adam optimizer
model.compile(loss='mse', optimizer='adam')

# 20% of data will be used for validation
# Training using 3 epochs
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

# Saving model
model.save('model.h5')