import cv2
import os
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from collections import defaultdict, deque
import time

# ============================
# CONFIGURATION
# ============================

DATASET_PATH = "data/registered_faces"

DETECTION_SIZE = (640, 640)
RECOGNITION_THRESHOLD = 0.45

FRAME_SKIP = 3
STABILITY_FRAMES = 5

# ============================
# ENGINE
# ============================

class InsightFaceAttendance:

    def __init__(self):

        print("Loading InsightFace model...")

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=['CPUExecutionProvider']
        )

        self.app.prepare(
            ctx_id=0,
            det_size=DETECTION_SIZE
        )

        print("Loading face database...")
        self.embeddings_db = {}
        self.load_database()

        self.frame_count = 0

        # Stabilization memory
        self.face_memory = defaultdict(lambda: deque(maxlen=STABILITY_FRAMES))

        self.last_results = []

        print("System ready.")

    # ============================
    # LOAD DATABASE
    # ============================

    def load_database(self):

        for person in os.listdir(DATASET_PATH):

            person_path = os.path.join(DATASET_PATH, person)

            if not os.path.isdir(person_path):
                continue

            self.embeddings_db[person] = []

            for file in os.listdir(person_path):

                if file.startswith("."):
                    continue

                path = os.path.join(person_path, file)

                img = cv2.imread(path)

                if img is None:
                    continue

                faces = self.app.get(img)

                if len(faces) == 0:
                    continue

                embedding = faces[0].embedding

                self.embeddings_db[person].append(embedding)

        print("Database loaded.")

    # ============================
    # COSINE DISTANCE
    # ============================

    def cosine_distance(self, a, b):

        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # ============================
    # RECOGNITION
    # ============================

    def recognize_frame(self, frame):

        self.frame_count += 1

        # Skip frames for performance
        if self.frame_count % FRAME_SKIP != 0:

            return self.last_results

        small = cv2.resize(frame, (640, 480))

        faces = self.app.get(small)

        results = []

        for face in faces:

            emb = face.embedding

            best_name = "Unknown"
            best_distance = 1.0

            for person in self.embeddings_db:

                for db_emb in self.embeddings_db[person]:

                    dist = self.cosine_distance(emb, db_emb)

                    if dist < best_distance:

                        best_distance = dist
                        best_name = person

            confidence = (1 - best_distance) * 100

            if best_distance > RECOGNITION_THRESHOLD:

                best_name = "Unknown"

            # Stabilization
            key = tuple(face.bbox.astype(int))

            self.face_memory[key].append(best_name)

            stable_name = max(
                set(self.face_memory[key]),
                key=self.face_memory[key].count
            )

            bbox = face.bbox.astype(int)

            results.append({

                "name": stable_name,
                "confidence": confidence,
                "bbox": bbox

            })

        self.last_results = results

        return results

# ============================
# CAMERA RUNNER
# ============================

def run_camera():

    engine = InsightFaceAttendance()

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Camera started.")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        results = engine.recognize_frame(frame)

        for face in results:

            x1, y1, x2, y2 = face["bbox"]

            name = face["name"]

            conf = face["confidence"]

            color = (0,255,0) if name != "Unknown" else (0,0,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            cv2.putText(
                frame,
                f"{name} ({conf:.1f}%)",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ============================

if __name__ == "__main__":
    run_camera()
