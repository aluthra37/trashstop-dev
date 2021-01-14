import cv2
import numpy as np
from PIL import Image
from keras import models
import webbrowser

size = 128
#take from labels.txt
classifications = ['Cardboard','Glass','Metal','Nothing','Paper','Plastic','Landfill']
#Load the saved model
model = models.load_model('model-adapted.h5')
video = cv2.VideoCapture(0)

while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into 128x128 because we trained the model with this image size.
        im = im.resize((size,size))
        img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3
        img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict method on model to predict object on the image
        prediction = model.predict(img_array)

        print("Maximum Probability: ",np.max(prediction[0], axis=-1))
        predicted_class = np.argmax(prediction[0], axis=-1)
        print(classifications[predicted_class])

        #show respective page
        showPage(classifications[predicted_class])

        cv2.imshow("Capturing", frame)

        #exit
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()


def showPage(id):
        '''
        if id=="Nothing":
                #show nothing page: livestream
        elif id=="Cardboard":
                #show cardboard page
        elif id=="Glass":
                #show cardboard page
        elif id=="Metal":
                #show cardboard page
        elif id=="Paper":
                #show cardboard page
        elif id=="Plastic":
                #show cardboard page
        elif id=="Landfill":
                #show cardboard page
        '''
