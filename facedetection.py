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
        cv2.imwrite(image_path.replace('dataset_split', 'dataset_split_face').split('.jpg')[0]+'_{}.jpg'.format(count), img_face)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1

    cv2.destroyAllWindows()

image_path = 'dataset_split'

for whatelse in os.listdir(image_path):
    whatelse_path = os.path.join(image_path, whatelse)
    for sub_whatelse in os.listdir(whatelse_path):
        sub_sub_whatelse =os.path.join(whatelse_path, sub_whatelse)
        for sub_sub_sub_whatelse in os.listdir(sub_sub_whatelse):
            img_path = os.path.join(sub_sub_whatelse, sub_sub_sub_whatelse)

            if not os.path.isdir(sub_sub_whatelse.replace('dataset_split', 'dataset_split_face')):
                os.makedirs(sub_sub_whatelse.replace('dataset_split', 'dataset_split_face'))
                print(sub_sub_whatelse.replace('dataset_split', 'dataset_split_face'))

            if img_path.endswith('.jpg'):
                getFaces(img_path)






