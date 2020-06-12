''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18

'''

import cv2
import os

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while True:
    ret, img = cap.read()
    img = cv2.flip(img, -1)
    # 获取图片尺寸并计算图片中心点
    (h, w) = img.shape[:2]
    center = (w/2, h/2)

    # 将图像旋转180度
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    img = rotated
    cv2.imshow('image', img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,

        scaleFactor=1.2,
        minNeighbors=5
        ,
        minSize=(20, 20)
    )
    print(count)
    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("./dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        print(count)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 1000: # Take 30 face sample and stop video
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()


