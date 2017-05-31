import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import random

import cv2

import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
#from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D, Cropping2D, Dropout
from keras.regularizers import l2


def crop_image(image):
    cropped_img = cv2.resize(image[50:140, :], (320, 80))
    return cropped_img;


def random_brightness(image):
    # apply random brightness, produce darker transformation 
    
    # Convert 2 HSV colorspace from RGB colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Generate new random brightness
    rand = random.uniform(0.3,1.0)
    hsv[:,:,2] = rand*hsv[:,:,2]
    # Convert back to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return new_img 

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    center_recover_offset = 0.20
    
    while 1:# Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]
            
            images = []
            angles = []
            
            for batch_sample in batch_samples:
                center_image_path = './data/IMG/' + batch_sample[0].split('/')[-1]
                left_image_path = './data/IMG/' + batch_sample[1].split('/')[-1]
                right_image_path = './data/IMG/' + batch_sample[2].split('/')[-1]
                  
                #center_image = cv2.imread(center_image_path)
                center_image = mpimg.imread(center_image_path)
                center_image = crop_image(center_image) # trim image to only see section with road
                center_image = random_brightness(center_image)
                
                left_image = mpimg.imread(left_image_path)
                left_image = crop_image(left_image)
                left_image = random_brightness(left_image)
                
                right_image = mpimg.imread(right_image_path)
                right_image = crop_image(right_image)
                right_image = random_brightness(right_image)
                
                center_angle = float(batch_sample[3])
                left_angle = center_angle + center_recover_offset
                right_angle = center_angle - center_recover_offset
                
                images.append(center_image)
                images.append(cv2.flip(center_image, 1))
                
                angles.append(center_angle)
                angles.append(-1.0 * center_angle) #flipped angle
                
                
                images.append(left_image)
                images.append(cv2.flip(left_image, 1))
                
                angles.append(left_angle)
                angles.append(-1 * left_angle) #flipped angle
                
                
                images.append(right_image)
                images.append(cv2.flip(right_image, 1))
                angles.append(right_angle)
                angles.append(-1 * right_angle)#flipped angle

            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)

# data collection and preparation
samples = []
data_file_path = './data/driving_log.csv'

with open(data_file_path) as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        samples.append(line)
        
# training data sets         
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

batch_size = 128

# build model base on nvidia's architecture
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (80, 320, 3)))

model.add(Convolution2D(24,5,5,border_mode='valid', subsample=(2,2), activation='relu', W_regularizer = l2(0.001)))
model.add(Convolution2D(36,5,5,border_mode='valid', subsample=(2,2), activation='relu',  W_regularizer = l2(0.001)))
model.add(Convolution2D(48,5,5, border_mode='valid', subsample=(2,2), activation='relu',  W_regularizer = l2(0.001)))
model.add(Dropout(0.2))

model.add(Convolution2D(64,3,3,border_mode='same', subsample=(2,2), activation='relu',  W_regularizer = l2(0.001)))
model.add(Convolution2D(64,3,3,border_mode='valid', subsample=(2,2), activation='relu',  W_regularizer = l2(0.001)))

model.add(Flatten())
model.add(Dense(100, W_regularizer = l2(0.001)))
model.add(Dense(50, W_regularizer = l2(0.001)))
model.add(Dense(10,  W_regularizer = l2(0.001)))
model.add(Dense(1,W_regularizer = l2(0.001)))

model.compile(loss='mse', optimizer='adam')

# train model
model.fit_generator(train_generator, samples_per_epoch = len(train_samples)//batch_size,\
                    validation_data = validation_generator, nb_val_samples = len(validation_samples)//batch_size, nb_epoch = 5)

# save model
model.save('model.h5')

print('Model saved!');