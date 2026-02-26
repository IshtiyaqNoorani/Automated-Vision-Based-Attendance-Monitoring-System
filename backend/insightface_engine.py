import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

DATASET_PATH = "data/registered_faces"
RECOGNITION_THRESHOLD = 0.45

DETECT_EVERY_N_FRAMES = 5


class FaceEngine:

    def __init__(self):

        print("Loading InsightFace model...")

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=['CPUExecutionProvider']
        )

        self.app.prepare(ctx_id=0, det_size=(640,640))

        self.embeddings_db = {}
        self.load_database()

        self.trackers = []
        self.tracker_names = []

        self.frame_count = 0

        print("Engine ready.")

    def load_database(self):

        for person in os.listdir(DATASET_PATH):

            person_path = os.path.join(DATASET_PATH, person)

            if not os.path.isdir(person_path):
                continue

            self.embeddings_db[person] = []

            for file in os.listdir(person_path):

                if file.startswith("."):
                    continue

                img = cv2.imread(os.path.join(person_path,file))

                faces = self.app.get(img)

                if len(faces):

                    self.embeddings_db[person].append(
                        faces[0].embedding
                    )

    def cosine_distance(self,a,b):

        return 1 - np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

    def recognize(self, frame):

        self.frame_count += 1

        results = []

        # update trackers every frame
        new_trackers = []
        new_names = []

        for tracker, name in zip(self.trackers, self.tracker_names):

            success, box = tracker.update(frame)

            if success:

                x,y,w,h = [int(v) for v in box]

                results.append({
                    "name": name,
                    "box": (x,y,x+w,y+h)
                })

                new_trackers.append(tracker)
                new_names.append(name)

        self.trackers = new_trackers
        self.tracker_names = new_names

        # run recognition occasionally
        if self.frame_count % DETECT_EVERY_N_FRAMES == 0:

            faces = self.app.get(frame)

            for face in faces:

                embedding = face.embedding

                best_name = "Unknown"
                best_dist = 1.0

                for person in self.embeddings_db:

                    for db_emb in self.embeddings_db[person]:

                        dist = self.cosine_distance(
                            embedding, db_emb
                        )

                        if dist < best_dist:

                            best_dist = dist
                            best_name = person

                if best_dist > RECOGNITION_THRESHOLD:

                    best_name = "Unknown"

                x1,y1,x2,y2 = face.bbox.astype(int)

                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame,(x1,y1,x2-x1,y2-y1))

                self.trackers.append(tracker)
                self.tracker_names.append(best_name)

        return results


def run_camera():

    engine = FaceEngine()

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    while True:

        ret, frame = cap.read()

        frame = cv2.flip(frame,1)

        results = engine.recognize(frame)

        for face in results:

            x1,y1,x2,y2 = face["box"]

            name = face["name"]

            color = (0,255,0) if name!="Unknown" else (0,0,255)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

            cv2.putText(
                frame,name,(x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2
            )

        cv2.imshow("Attendance System",frame)

        if cv2.waitKey(1)==ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    run_camera()
