import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

DATASET_PATH = "data/registered_faces"
RECOGNITION_THRESHOLD = 0.45


class InsightFaceEngine:

    def __init__(self):

        print("Initializing InsightFace...")

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=['CPUExecutionProvider']
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

        self.embeddings_db = {}

        self.load_database()

        print("Engine ready.")

    # =========================

    def load_database(self):

        for person in os.listdir(DATASET_PATH):

            person_path = os.path.join(DATASET_PATH, person)

            if not os.path.isdir(person_path):
                continue

            self.embeddings_db[person] = []

            for file in os.listdir(person_path):

                if file.startswith("."):
                    continue

                img_path = os.path.join(person_path, file)

                img = cv2.imread(img_path)

                if img is None:
                    continue

                faces = self.app.get(img)

                if len(faces) == 0:
                    continue

                embedding = faces[0].embedding

                self.embeddings_db[person].append(embedding)

        print("Database loaded.")

    # =========================

    def cosine_distance(self, a, b):

        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # =========================

    def recognize(self, frame):

        faces = self.app.get(frame)

        results = []

        for face in faces:

            embedding = face.embedding

            best_name = "Unknown"
            best_distance = 1.0

            for person in self.embeddings_db:

                for db_embedding in self.embeddings_db[person]:

                    dist = self.cosine_distance(
                        embedding,
                        db_embedding
                    )

                    if dist < best_distance:

                        best_distance = dist
                        best_name = person

            confidence = (1 - best_distance) * 100

            if best_distance > RECOGNITION_THRESHOLD:

                best_name = "Unknown"

            bbox = face.bbox.astype(int)

            results.append({

                "name": best_name,
                "confidence": confidence,
                "box": bbox

            })

        return results


# =========================

def run_camera():

    engine = InsightFaceEngine()

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        results = engine.recognize(frame)

        for face in results:

            x1, y1, x2, y2 = face["box"]

            name = face["name"]

            confidence = face["confidence"]

            color = (0,255,0) if name != "Unknown" else (0,0,255)

            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                color,
                2
            )

            cv2.putText(
                frame,
                f"{name} ({confidence:.1f}%)",
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


if __name__ == "__main__":

    run_camera()
