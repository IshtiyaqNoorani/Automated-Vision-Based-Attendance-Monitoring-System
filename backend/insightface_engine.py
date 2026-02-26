import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

DATASET_PATH = "data/registered_faces"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

RECOGNITION_THRESHOLD = 0.42
IOU_THRESHOLD = 0.4

DETECTION_INTERVAL = 12


class TrackFace:
    def __init__(self, tracker, name, embedding, box):
        self.tracker = tracker
        self.name = name
        self.embedding = embedding
        self.box = box


def cosine_distance(a, b):
    return 1 - np.dot(a, b)


def compute_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0

    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


class EnterpriseFaceEngine:

    def __init__(self):

        print("Loading buffalo_l model...")

        self.app = FaceAnalysis(name="buffalo_l",
                                providers=["CPUExecutionProvider"])

        self.app.prepare(ctx_id=0, det_size=(640,640))

        self.database = {}
        self.load_database()

        self.trackers = []
        self.frame_count = 0


    def load_database(self):

        print("Loading database...")

        for person in os.listdir(DATASET_PATH):

            path = os.path.join(DATASET_PATH, person)

            if not os.path.isdir(path):
                continue

            embeddings = []

            for file in os.listdir(path):

                if file.startswith("."):
                    continue

                img = cv2.imread(os.path.join(path,file))

                faces = self.app.get(img)

                if len(faces)==0:
                    continue

                emb = faces[0].embedding
                emb = emb / np.linalg.norm(emb)

                embeddings.append(emb)

            if embeddings:
                self.database[person] = embeddings

        print("Loaded:", list(self.database.keys()))


    def match(self, embedding):

        best_name = "Unknown"
        best_dist = 1.0

        for person in self.database:

            for emb in self.database[person]:

                dist = cosine_distance(embedding, emb)

                if dist < best_dist:
                    best_dist = dist
                    best_name = person

        if best_dist > RECOGNITION_THRESHOLD:
            return "Unknown"

        return best_name


    def is_duplicate_box(self, new_box):

        for face in self.trackers:

            if compute_iou(face.box, new_box) > IOU_THRESHOLD:
                return True

        return False


    def update_trackers(self, frame):

        active = []
        results = []

        for face in self.trackers:

            ok, box = face.tracker.update(frame)

            if ok:

                x,y,w,h = map(int, box)

                face.box = (x,y,x+w,y+h)

                active.append(face)

                results.append({
                    "name": face.name,
                    "box": face.box
                })

        self.trackers = active

        return results


    def detect_new_faces(self, frame):

        faces = self.app.get(frame)

        for face in faces:

            emb = face.embedding
            emb = emb / np.linalg.norm(emb)

            box = tuple(face.bbox.astype(int))

            if self.is_duplicate_box(box):
                continue

            name = self.match(emb)

            tracker = cv2.TrackerKCF_create()
            tracker.init(frame,
                         (box[0],box[1],box[2]-box[0],box[3]-box[1]))

            self.trackers.append(
                TrackFace(tracker, name, emb, box)
            )


    def process(self, frame):

        self.frame_count += 1

        results = self.update_trackers(frame)

        if self.frame_count % DETECTION_INTERVAL == 0:
            self.detect_new_faces(frame)

        return results


def run():

    engine = EnterpriseFaceEngine()

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame,1)

        results = engine.process(frame)

        for r in results:

            x1,y1,x2,y2 = r["box"]

            color = (0,255,0) if r["name"]!="Unknown" else (0,0,255)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

            cv2.putText(frame,r["name"],
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,color,2)

        cv2.imshow("Enterprise Attendance System", frame)

        if cv2.waitKey(1)==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    run()
