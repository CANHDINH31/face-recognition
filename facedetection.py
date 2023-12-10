import cv2
import os

face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

def getFaces(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(img_gray, 1.3, 5)
    count = 0
    for (x, y, w, h) in faces:
        img_face = cv2.resize(img[y + 3: y + h - 3, x + 3: x + w - 3], (64, 64))
        cv2.imwrite(image_path.replace('image', 'image_face').split('.jpg')[0]+'_{}.jpg'.format(count), img_face)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1

    cv2.destroyAllWindows()

image_path = 'image'

for whatelse in os.listdir(image_path):
    whatelse_path = os.path.join(image_path, whatelse)
    for sub_whatelse in os.listdir(whatelse_path):
        img_path =os.path.join(whatelse_path, sub_whatelse)
        if not os.path.isdir(whatelse_path.replace('image', 'image_face')):
            os.makedirs(whatelse_path.replace('image', 'image_face'))
            print(whatelse_path.replace('image', 'image_face'))

        if img_path.endswith('.jpg'):
            getFaces(img_path)






