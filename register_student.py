import cv2
import os
import time
import numpy as np
from insightface.app import FaceAnalysis

DATASET_PATH = "data/registered_faces"
NUM_IMAGES = 25

BLUR_THRESHOLD = 80
MIN_FACE_SIZE = 120

app = FaceAnalysis(name="buffalo_l",
                   providers=["CPUExecutionProvider"])

app.prepare(ctx_id=0, det_size=(640,640))

student = input("Enter student name: ")

path = os.path.join(DATASET_PATH, student)
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

while count < NUM_IMAGES:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    faces = app.get(frame)

    if len(faces):

        face = faces[0]

        x1,y1,x2,y2 = face.bbox.astype(int)

        face_img = frame[y1:y2,x1:x2]

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        blur = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur < BLUR_THRESHOLD:
            continue

        if face_img.shape[0] < MIN_FACE_SIZE:
            continue

        filename = f"{student}_{count}.jpg"

        cv2.imwrite(os.path.join(path, filename), frame)

        count += 1

        print("Saved", count)

        time.sleep(0.4)

    cv2.imshow("Register", frame)

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
