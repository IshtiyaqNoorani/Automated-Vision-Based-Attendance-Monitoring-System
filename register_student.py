import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

DATASET_DIR = "data/registered_faces"
NUM_IMAGES = 25

print("Loading InsightFace buffalo_l for registration...")

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640,640))

print("Model ready.")

name = input("Enter student roll and name (example: 2408912_Ishtiyaq): ")

student_dir = os.path.join(DATASET_DIR, name)
os.makedirs(student_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

print("Press SPACE to capture image")
print("Press ESC to exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame,1)

    faces = app.get(frame)

    display = frame.copy()

    for face in faces:

        x1,y1,x2,y2 = map(int, face.bbox)

        cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow("Register Student", display)

    key = cv2.waitKey(1)

    if key == 27:
        break

    if key == 32:

        if len(faces)==0:
            print("No face detected")
            continue

        filename = f"{name}_{count}.jpg"

        path = os.path.join(student_dir, filename)

        cv2.imwrite(path, frame)

        print("Saved:", path)

        count += 1

        if count >= NUM_IMAGES:
            break

cap.release()
cv2.destroyAllWindows()

print("Registration complete.")
