import sys
import os
import numpy as np
import cv2
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Conv2D, Flatten, MaxPooling2D,Dense,Dropout,SpatialDropout2D, BatchNormalization, MaxPool2D, Input
from keras.models  import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import random,os,glob
import matplotlib.pyplot as plt

argvs = sys.argv
argc = len(argvs)
epochs = int(argvs[1])
select = argvs[2]

size = 110
batch_size = 64 #was 64, 32
validation_split = 0.3

if(select=='garbage'):
  dir_path = './classification/garbage-classification/garbage-classification'
  num_images_train = 2927-(2927*validation_split)
  num_images_val = 2927*validation_split
  num_classes = 7
  print("Garbage Dataset Selected")
elif(select=='cifar'):
  dir_path = './data/dataset/train'
  num_images_train = 50000-(50000*validation_split)
  num_images_val = 50000*validation_split
  num_classes = 10
  print("CIFAR-10 Dataset Selected")
elif(select=='cifar-c'):
  dir_path = './data-condensed/dataset/train'
  num_images_train = 4010-(4010*validation_split)
  num_images_val = 4010*validation_split
  num_classes = 10
  print("CIFAR-10 Condensed Dataset Selected")

img_list = glob.glob(os.path.join(dir_path, '*/*.jpg'))
num_images = len(img_list)
print(num_images)

train=ImageDataGenerator(horizontal_flip=False,
                         vertical_flip=False,
                         validation_split=validation_split,
                         rescale=1./255,
                         shear_range = 0.2,
                         zoom_range = 0.2,
                         )

test=ImageDataGenerator(rescale=1/255,
                        validation_split=validation_split)

train_generator=train.flow_from_directory(dir_path,
                                          target_size=(size,size),
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          subset='training')


test_generator=test.flow_from_directory(dir_path,
                                        target_size=(size,size),
                                        batch_size=batch_size,
                                        class_mode='categorical',
                                        subset='validation')

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
#print(labels)
for image_batch, label_batch in train_generator:
  break
image_batch.shape, label_batch.shape
#print(image_batch.shape, label_batch.shape)
Labels = '\n'.join(sorted(train_generator.class_indices.keys()))
with open('labels.txt', 'w') as f:
  f.write(Labels)

'''
model=Sequential()
#Convolution blocks
model.add(Conv2D(32,(3,3), padding='same',input_shape=(size,size,3),activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64,(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(32,(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
#Classification layers
model.add(Flatten())
model.add(Dense(128,activation='relu')) #was 64
model.add(Dropout(0.2)) #0.2 better than 0.4
model.add(Dense(64,activation='relu'))#was 32
model.add(Dropout(0.2)) #0.2 better acc then 0.4, better loss, 0.1 better than 0.2 in acc/loss
model.add(Dense(512,activation='relu'))#was 32
model.add(Dropout(0.1))
model.add(Dense(num_classes,activation='softmax'))
'''

model = Sequential()
model.add(Conv2D(32, kernel_size=3,input_shape=(size,size,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(32, kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.4))
model.add(Conv2D(64, kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes,activation='softmax'))

'''
model = Sequential()
model.add(Input(shape=(size,size,3)))
model.add(Conv2D(32, kernel_size=3,input_shape=(size,size,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(32, kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.4))
model.add(Conv2D(64, kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes,activation='softmax'))
'''

filepath="trained_model.h5"
checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
callbacks_list = [checkpoint1]

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc']) # RMS PROP - No accuracy

#es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

history = model.fit(train_generator,
                              epochs=epochs,
                              steps_per_epoch=num_images_train//batch_size,
                              validation_data=test_generator,
                              validation_steps=num_images_val//batch_size,
                              workers = 4,
                              callbacks=callbacks_list)
#41 epoch - 75% #73- 76.9%
#78 epoch - 80%

plt.figure()
plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# summarize history for loss
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
