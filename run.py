import cv2
from keras_facenet import FaceNet
from sklearn.neighbors import RadiusNeighborsClassifier
import os
facenet = FaceNet()


rnc = RadiusNeighborsClassifier(radius=0.7, outlier_label='Unknown')
X = []
y = []

for entry in os.listdir('data'):
    for img in os.listdir(os.path.join('data', entry)):
        detections = facenet.extract(os.path.join('data', entry, img))
        if len(detections)==0: continue
        X.append(detections[0]['embedding'])
        y.append(entry)

rnc.fit(X, y)


def detectAndDisplay(frame):
    detections = facenet.extract(frame)

    for detection in detections:      
        box = detection['box']
        frame = cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 0, 255), 2)
        predtiction = rnc.predict([detection['embedding']])
        frame = cv2.putText(frame, predtiction[0], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.imshow('Capture - Face detection', frame)

webcam = cv2.VideoCapture(0)
check, frame = webcam.read()

if not webcam.isOpened:
    print('No webcam detected!')
    exit()

while True:
    check, frame = webcam.read()
    if not check:
        print('webcam disconnected')
        exit()
    
    detectAndDisplay(frame)

    # cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break