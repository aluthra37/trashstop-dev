from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.models  import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D,Dense,Dropout,SpatialDropout2D, Activation, BatchNormalization, MaxPool2D
import numpy as np
import cv2
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing.image import img_to_array, load_img, array_to_img
import random,os,glob
import matplotlib.pyplot as plt
from keras import initializers

#location of all trash images
dir_path = './classification/garbage-classification/garbage-classification'

img_list = glob.glob(os.path.join(dir_path, '*/*.jpg'))

print(len(img_list))

#training and testing images are 128x128
size=32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1.255, validation_split=0.1)

train_generator = train_datagen.flow_from_directory(dir_path,target_size=(size,size),batch_size=32, class_mode='categorical', subset='training')
validation_generator = validation_datagen.flow_from_directory(dir_path, target_size=(size,size), batch_size=32, class_mode='categorical', subset='validation')

#classification map
labels = (train_generator.class_indices)
#flipping keys and values in classification map
labels = dict((v,k) for k,v in labels.items())
print(labels)

print (train_generator.class_indices)

Labels = '\n'.join(sorted(train_generator.class_indices.keys()))

print(Labels)


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(size, size ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(7, activation='softmax')])


"""
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
model.add(layers.Dense(7, activation='sigmoid'))
"""
"""
model=Sequential()
#Convolution blocks

model.add(Conv2D(32,(3,3), padding='same',input_shape=(size,size,3),activation='relu'))
model.add(MaxPooling2D(2,2))
#model.add(SpatialDropout2D(0.5)) # No accuracy

model.add(Conv2D(64,(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(SpatialDropout2D(0.4))

model.add(Conv2D(32,(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))

#Classification layers
model.add(Flatten())

model.add(Dense(32,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32,activation='relu'))

model.add(Dropout(0.4))
model.add(Dense(7,activation='softmax'))
"""
"""
# Creating a Sequential model
model= Sequential()
model.add(Conv2D(kernel_size=(3,3), filters=32, activation='tanh', input_shape=(size,size,3,)))
model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))

model.add(Flatten())

model.add(Dense(20,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(7,activation = 'softmax'))
"""

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(train_generator, epochs=15, steps_per_epoch=64, validation_data=validation_generator, validation_steps=7, workers=4)

model.save('model-adapted.h5')

print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
