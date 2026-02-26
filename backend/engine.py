import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

DATASET_DIR = "data/registered_faces"

SIM_THRESHOLD = 0.45

class Engine:

    def __init__(self):

        print("Loading InsightFace buffalo_l model...")

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(ctx_id=0, det_size=(640, 640))

        print("Model ready")

        self.embeddings = []
        self.names = []

        self.load_faces()

    def load_faces(self):

        print("Loading registered faces...")

        for person in os.listdir(DATASET_DIR):

            person_dir = os.path.join(DATASET_DIR, person)

            if not os.path.isdir(person_dir):
                continue

            for file in os.listdir(person_dir):

                if not file.lower().endswith((".jpg",".jpeg",".png")):
                    continue

                path = os.path.join(person_dir, file)

                img = cv2.imread(path)

                if img is None:
                    continue

                faces = self.app.get(img)

                if len(faces) == 0:
                    continue

                emb = faces[0].embedding

                emb = emb / np.linalg.norm(emb)

                self.embeddings.append(emb)
                self.names.append(person)

        self.embeddings = np.array(self.embeddings)

        print("Loaded", len(self.names), "faces")

    def match(self, emb):

        if len(self.embeddings) == 0:
            return "Unknown", 0

        emb = emb / np.linalg.norm(emb)

        sims = np.dot(self.embeddings, emb)

        idx = np.argmax(sims)

        if sims[idx] > SIM_THRESHOLD:

            return self.names[idx], sims[idx]

        return "Unknown", sims[idx]


def remove_duplicates(faces):

    if len(faces) == 0:
        return []

    faces = sorted(faces, key=lambda f: f.det_score, reverse=True)

    filtered = []

    for face in faces:

        x1,y1,x2,y2 = map(int, face.bbox)

        keep = True

        for f in filtered:

            fx1,fy1,fx2,fy2 = map(int, f.bbox)

            cx = (x1+x2)//2
            cy = (y1+y2)//2

            if fx1 < cx < fx2 and fy1 < cy < fy2:

                keep = False
                break

        if keep:
            filtered.append(face)

    return filtered


def run():

    engine = Engine()

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    print("Camera started")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # FIX inverted preview
        frame = cv2.flip(frame,1)

        faces = engine.app.get(frame)

        faces = remove_duplicates(faces)

        for face in faces:

            emb = face.embedding

            name,score = engine.match(emb)

            x1,y1,x2,y2 = map(int, face.bbox)

            color = (0,255,0) if name!="Unknown" else (0,0,255)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

            cv2.putText(
                frame,
                f"{name} {score:.2f}",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow("Attendance System",frame)

        if cv2.waitKey(1)==27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    run()
