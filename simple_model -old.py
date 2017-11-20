import csv
import cv2
import numpy as np

lines = []
with open('data_trial4/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        

images = []
measurements = []
for line in lines:
	source_path = line[0]
	image = cv2.imread(source_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)
	
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D

#image_flipped = np.fliplr(image)
#measurement_flipped = -measurement

model = Sequential()
#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(32, 3, 3,
                 activation='relu',
                 input_shape=(160,320,3)))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))


#model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.49, shuffle=True, nb_epoch=5)

model.save('model.h5')