import cv2
import os

face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')

import time

cam = cv2.VideoCapture(0)
# cam = cv2.VideoCapture('video/messi.mp4');

count = 0
while True:
    OK, frame = cam.read()
    faces = face_detector.detectMultiScale(frame, 1.3, 5)
    # time.sleep(0.3)
    for (x, y, w, h) in faces:
        roi = cv2.resize(frame[y+2:y+h-2, x+2:x+w-2], (100,100))
        cv2.imwrite('image_face_video/messi/messi{}.jpg'.format(count), roi)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (128,255,50), 1)
        count += 1

    cv2.imshow("FRAME", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cam.release()
cv2.destroyAllWindows()
