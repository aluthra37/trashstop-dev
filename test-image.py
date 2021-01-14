#image recognition and processing
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import cv2
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Conv2D, Flatten, MaxPooling2D,Dense,Dropout,SpatialDropout2D
from keras.models  import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import random,os,glob
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_path = "./classification/Garbage classification/Garbage classification/plastic/plastic44.jpg"

img=mpimg.imread(img_path)
imgplot = plt.imshow(img)
plt.show()

size = 128

img = image.load_img(img_path, target_size=(size, size))
img = image.img_to_array(img, dtype=np.uint8)

plt.title("Loaded Image")
plt.axis('off')
plt.imshow(img.squeeze())


loaded_model = load_model("model-adapted.h5")

p=loaded_model.predict(img[np.newaxis, ...])

print("Maximum Probability: ",np.max(p[0], axis=-1))
predicted_class = np.argmax(p[0], axis=-1)

if predicted_class == 0:
	item = "Cardboard"
elif predicted_class == 1:
	item = "Glass"
elif predicted_class == 2:
	item = "Metal"
elif predicted_class == 3:
	item = "Paper"
elif predicted_class == 4:
	item = "Plastic"
else:
	item = "Landfill"

print("Classified:", item)
print(predicted_class)
