import csv
import cv2
import numpy as np

# reading the data
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line) # store the csv files

# For the purpose of the project using only Center Image and Steering Angle
images = []
measurements = []

# In order to improve the distribution of images, i.e steering angle very close to zero were removed
# Correction factor was introduced for images from left and right camera
for line in lines[1:]:
    measurement = abs(round(float(line[3]),4))
    if measurement > 0.08:
        for i in range(3):
            source_path = line[i]
            tokens = source_path.split('/')
            filename = tokens[-1]
            local_path = "./data/IMG/" + filename
            image = cv2.imread(local_path)
            images.append(image)
        correction = 0.2
        measurement = float(line[3])
        measurements.append(measurement)
        measurements.append(measurement+correction)
        measurements.append(measurement-correction)

# Due to left Biased turns and ading flipped images
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image,1)
    flipped_measurement = measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

# Creating the training dataset
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# parameters
batch_size = 128
n_epoch = 10 # The number of iterations


## MODELLING
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D


# Model based on Nvidia deep learning architecture with dropouts(to stop overfitting)
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (1,1)),input_shape=(160, 320, 3))) # cropping the top and bottom part of the image as it does not provide any usable information
model.add(Lambda(lambda x: x / 255.0 - 0.5)) # normalizing the image
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(48, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1)) # single output node to predict the steering angle

# compile
model.compile(optimizer="adam", loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=n_epoch, batch_size=batch_size)
model.save('model.h5')


