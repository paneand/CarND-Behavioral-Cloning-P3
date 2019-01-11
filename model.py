import os
from pathlib import Path
import csv
import json
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from keras.models import Sequential, model_from_json
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D

########## Hyperparmeters ##########

model_h5 = 'model.h5'
model_json = 'model.json'
weights_h5 = 'weights.h5'



########## LOADING THE DATA ##########

samples = []


def add_to_samples(csv_filepath):
    with open(csv_filepath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

add_to_samples('./data/driving_log.csv')
add_to_samples('./data/driving_log_straight.csv')
add_to_samples('./data/driving_log_inversed.csv')
add_to_samples('./data/driving_log_last_curve.csv')

from sklearn.model_selection import train_test_split

samples = samples[1:]
shuffle(samples)

(train_samples, validation_samples) = train_test_split(samples,
        test_size=0.2)




## 2. Data Preprocessing functions


''' 
Since the dataset is huge, a generator needs to be used. 
Only the values from each iteration will be loaded in memory. 
This function will return a generator ("yield" used instead of "return") to iterate over the dataset. 
'''
def generator(samples, batch_size=32):
    num_samples = len(samples)
    batch_size = batch_size // 4        # // In python 3 means integer division (rounded up)
    while 1:                            # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # Retrieve the images
                split_char = '/'
                if batch_sample[0].find(split_char) == -1:
                    split_char = '\\'
                name = './data/IMG/' \
                    + batch_sample[0].split(split_char)[-1]
                left_name = './data/IMG/' \
                    + batch_sample[1].split(split_char)[-1]
                right_name = './data/IMG/' \
                    + batch_sample[2].split(split_char)[-1]
                
                center_image = cv2.imread(name)             # Image taken from the centered camera on the car
                image_flipped = np.fliplr(center_image)     # Centered image, flipped
                left_image = cv2.imread(left_name)          # Image taken from the left camera on the car
                right_image = cv2.imread(right_name)        # Image taken from the right camera on the car

                center_angle = float(batch_sample[3])       # Real steering angle of the car 
                measurement_flipped = -center_angle         # Flipped steering angle of the car, to be used with the flipped image 

                """
                The left and right images are exploited using a correction parameter on the steering angle of the car.
                The left image captures more the left side of the road, giving the impression that the car is steering left.
                Therefore a positive (positive angles go in counterclock wise direction) correction on the angle is applied.
                Same is applied, with negative correction, on the right image
                """
                correction = 0.15  # this is a tunable parameter
                steering_left = center_angle + correction
                steering_right = center_angle - correction

                # Collect images and associated steering values
                images.append(center_image)
                angles.append(center_angle)
                images.append(image_flipped)
                angles.append(measurement_flipped)
                images.append(left_image)
                angles.append(steering_left)
                images.append(right_image)
                angles.append(steering_right)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

""" 
Since generator() returns a generator, the code inside of it will be executed only when there will be an iteration over 
train_generator or validation_generator  
"""
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print ('Train set size is :', len(train_samples))
print ('Validation set size is :', len(validation_samples))


########## MODEL ARCHITECTURE ##########

def resize_comma(image):
    import tensorflow as tf  # This import is required here otherwise the model cannot be loaded in drive.py
    return tf.image.resize_images(image, (40, 160))

def create_model():
	model = Sequential()
	model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160,
          320, 3)))		# Crop the image
	# Resize the data
	model.add(Lambda(resize_comma))
	model.add(Lambda(lambda x: x / 255.0 - 0.5))
	model.add(Conv2D(24, 5, 2, activation='relu'))
	model.add(Conv2D(36, 5, 2, activation='relu'))
	model.add(Conv2D(48, 5, 2, activation='relu'))
	model.add(Conv2D(64, 3, 1, activation='relu'))
	model.add(Conv2D(64, 3, 1, activation='relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	return model

########## MODEL LOADING ##########

if Path(model_json).is_file():
    with open(model_json, 'r') as json_file:
       model = model_from_json(json.load(json_file))
       model.load_weights(weights_h5)
       print("Loaded model from disk:")
       model.summary()
else:
	model = create_model()
	print("Created new model:")
	model.summary()


########## MODEL TRAINING ##########
#adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
#model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples * 4),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples * 4),
                    nb_epoch=3)

########## MODEL SAVING ##########

# Clean previous model and weights
if Path(model_json).is_file():
	os.remove(model_json)
	print('Removed old json model')
if Path(model_h5).is_file():
	os.remove(model_h5)
	print('Removed old h5 model')
if Path(weights_h5).is_file():
	os.remove(weights_h5)
	print('Removed old h5 weights')

with open(model_json, 'w') as json_file:
    json.dump(model.to_json(), json_file)
    print('Saved json model')
model.save(model_h5)
print('Saved h5 model')
model.save_weights(weights_h5)
print('Saved h5 weights')


			