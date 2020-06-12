from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import time
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')

# cap = cv2.VideoCapture('videos/礼让斑马线！齐齐哈尔城市文明的伤！.mp4')
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('../../data/TownCentreXVID.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)  # 25.0
print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print('共有', num_frames, '帧')  # 共有 2499.0 帧

frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print('高：', frame_height, '宽：', frame_width)  # 高： 480.0 宽： 640.0
# exit(0)


# 跳过多少帧
skips = 20

# loop over the image paths
# for imagePath in paths.list_images(args["images"]):
while cap.isOpened():

    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    # image = cv2.imread(imagePath)

    ret, frame = cap.read()
    img = frame
    current = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if current % skips != 0:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)#改进
    print("Detected ", len(faces), " face")

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            # 将每张检测到眼镜的照片存储
            # img = cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            # cv2.imwrite("wuyang.jpg",img)
    cv2.imshow('img', img)

    key = cv2.waitKey(delay=25)
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()