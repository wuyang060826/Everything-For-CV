''''
Training Multiple Faces stored on a DataBase:
	==> Each face should have a unique numeric integer ID as 1, 2, 3, etc
	==> LBPH computed model will be saved on trainer/ directory. (if it does not exist, pls create one)
	==> for using PIL, install pillow library with "pip install pillow"

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18

'''

import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')


# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        # PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        # # print(PIL_img)
        # img_numpy = np.array(PIL_img,'uint8')
        # print(img_numpy)
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        # print(id)
        # faces = detector.detectMultiScale(img_numpy)
        # print("Detected ", len(faces), " face")
        img = cv2.imread(imagePath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)#改进
        print("Detected ", len(faces), " face")
        for (x,y,w,h) in faces:
            faceSamples.append(gray[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
print(faces)
print(ids)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))