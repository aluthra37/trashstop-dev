from picamera.array import PiRGBArray # Generates a 3D RGB array
from picamera import PiCamera # Provides a Python interface for the RPi Camera Module
import time # Provides time-related functions
import cv2 # OpenCV library
import numpy as np
from PIL import Image
from keras import models
 
#size for capture
size = 110

#classifications array
classifications = ['Cardboard','Glass','Metal','Nothing','Paper','Plastic','Landfill']

#Load the saved model
model = models.load_model('trained_model.h5')

# Initialize the camera
camera = PiCamera()
 
# Set the camera resolution
camera.resolution = (size, size)
 
# Set the number of frames per second
camera.framerate = 32
 
# Generates a 3D RGB array and stores it in rawCapture
raw_capture = PiRGBArray(camera, size=(size, size))
 
# Wait a certain number of seconds to allow the camera time to warmup
time.sleep(0.1)
 
# Capture frames continuously from the camera
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
     
    # Grab the raw NumPy array representing the image
    image = np.asarray(frame.array)
 
    #resize
    np.resize(image,(size, size))

    img_array = np.array(image)

    #Our keras model used a 4D tensor, (images x height x width x channel)
    #So changing dimension 128x128x3 into 1x128x128x3
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    
    print("Maximum Probability: ",np.max(prediction[0], axis=-1))

    # Display the frame using OpenCV
    cv2.imshow("Frame", image)
     
    # Wait for keyPress for 1 millisecond
    key = cv2.waitKey(1) & 0xFF
     
    # Clear the stream in preparation for the next frame
    raw_capture.truncate(0)
     
    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
