import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

DATASET_DIR = "data/registered_faces"

SIMILARITY_THRESHOLD = 0.45
MIN_FACE_SIZE = 80


class InsightFaceEngine:

    def __init__(self):

        print("Loading InsightFace buffalo_l model...")

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640,640))

        print("Model loaded.")

        self.embeddings = []
        self.names = []

        self.load_database()


    def load_database(self):

        print("Loading face database...")

        self.embeddings.clear()
        self.names.clear()

        for person in os.listdir(DATASET_DIR):

            person_path = os.path.join(DATASET_DIR, person)

            if not os.path.isdir(person_path):
                continue

            for img_name in os.listdir(person_path):

                img_path = os.path.join(person_path, img_name)

                img = cv2.imread(img_path)

                if img is None:
                    continue

                faces = self.app.get(img)

                if len(faces) == 0:
                    continue

                embedding = faces[0].embedding

                self.embeddings.append(embedding)
                self.names.append(person)

        print("Database ready.")


    def recognize(self, frame):

        faces = self.app.get(frame)

        results = []

        for face in faces:

            x1,y1,x2,y2 = face.bbox.astype(int)

            if (x2-x1) < MIN_FACE_SIZE:
                continue

            embedding = face.embedding

            name = "Unknown"
            best_score = 0

            if len(self.embeddings) > 0:

                scores = cosine_similarity(
                    [embedding],
                    self.embeddings
                )[0]

                best_index = np.argmax(scores)
                best_score = scores[best_index]

                if best_score > SIMILARITY_THRESHOLD:

                    name = self.names[best_index]

            results.append({

                "name": name,
                "confidence": float(best_score),
                "box": (x1,y1,x2,y2)

            })

        return results


engine = InsightFaceEngine()


def run_camera():

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    print("Camera started.")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame,1)

        results = engine.recognize(frame)

        for r in results:

            x1,y1,x2,y2 = r["box"]
            name = r["name"]
            conf = r["confidence"]

            color = (0,255,0) if name!="Unknown" else (0,0,255)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

            label = f"{name} {conf:.2f}"

            cv2.putText(
                frame,
                label,
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow("Enterprise Attendance System", frame)

        if cv2.waitKey(1)==ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    run_camera()
