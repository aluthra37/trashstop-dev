from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import cv2
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing.image import img_to_array, load_img, array_to_img
import random,os,glob
import matplotlib.pyplot as plt

#location of all trash images
dir_path = './classification/Garbage classification/Garbage classification'

img_list = glob.glob(os.path.join(dir_path, '*/*.jpg'))

print(len(img_list))

#training and testing images are 128x128
size=128
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size,size,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.0003), loss='binary_crossentropy', metrics=['acc'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1.255)

train_generator = train_datagen.flow_from_directory('data/train',target_size=(size,size),batch_size=64, class_mode='binary')
validation_generator = validation_datagen.flow_from_directory('data/train/1', target_size=(size,size), batch_size=64, class_mode='binary')

model.fit_generator(train_generator, epochs=5, steps_per_epoch=63, validation_data=validation_generator, validation_steps=7, workers=4)

model.save('model.h5')
