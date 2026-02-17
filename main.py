import cv2
import os
import numpy as np
from collections import defaultdict
from deepface import DeepFace

from src.attendance import write_attendance


# =========================
# CONFIGURATION
# =========================

ATTENDANCE_MODE = "snapshot"   # "snapshot" or "live"

MODEL_NAME = "ArcFace"
DATASET_PATH = "data/registered_faces"

# Distance threshold for ArcFace (CRITICAL)
DISTANCE_THRESHOLD = 0.68

# Face processing
FACE_SIZE = (160, 160)
MIN_FACE_SIZE = 80

# Live mode settings
REQUIRED_DETECTIONS = 4


# =========================
# CAMERA CLASS
# =========================

class Camera:

    def __init__(self, cam_id=0):

        self.cap = cv2.VideoCapture(cam_id)

        # Higher resolution improves distant recognition
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        # Fix mirrored Mac webcam
        frame = cv2.flip(frame, 1)

        return frame

    def release(self):

        self.cap.release()


# =========================
# FAST FACE DETECTOR
# =========================

class FaceDetector:

    def __init__(self):

        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

    def detect_faces(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
        )

        return faces

    def draw_faces(self, frame, faces):

        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

        return frame


# =========================
# EMBEDDING DATABASE
# =========================

embeddings_db = {}


def load_embeddings():

    global embeddings_db

    print("Loading face embeddings...")

    embeddings_db = {}

    for person in os.listdir(DATASET_PATH):

        person_path = os.path.join(DATASET_PATH, person)

        if not os.path.isdir(person_path):
            continue

        embeddings_db[person] = []

        for img_name in os.listdir(person_path):

            img_path = os.path.join(person_path, img_name)

            try:

                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name=MODEL_NAME,
                    enforce_detection=False
                )[0]["embedding"]

                embeddings_db[person].append(
                    np.array(embedding)
                )

            except:
                pass

    print("Embeddings loaded successfully.")


# =========================
# FACE RECOGNITION
# =========================

def recognize_face(face_img):

    try:

        face_img = cv2.resize(
            face_img,
            FACE_SIZE,
            interpolation=cv2.INTER_CUBIC
        )

        embedding = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            enforce_detection=False
        )[0]["embedding"]

        embedding = np.array(embedding)

        best_match = None
        best_distance = 999

        for person in embeddings_db:

            for stored_embedding in embeddings_db[person]:

                distance = np.linalg.norm(
                    embedding - stored_embedding
                )

                if distance < best_distance:

                    best_distance = distance
                    best_match = person

        if best_distance < DISTANCE_THRESHOLD:

            confidence = (1 - best_distance) * 100

            return best_match, round(confidence, 2)

        return "Unknown", None

    except:

        return "Unknown", None


# =========================
# SNAPSHOT MODE (BEST ACCURACY)
# =========================

def snapshot_attendance(cam, detector):

    print("\nSnapshot mode started.")
    print("Press 'c' to capture attendance.")
    print("Press 'q' to quit.\n")

    while True:

        frame = cam.get_frame()

        if frame is None:
            continue

        faces = detector.detect_faces(frame)

        preview = frame.copy()

        preview = detector.draw_faces(preview, faces)

        cv2.imshow("Preview", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):

            cv2.destroyAllWindows()
            return set()

        if key == ord('c'):

            print("Processing snapshot...")

            present_students = set()

            for (x, y, w, h) in faces:

                face_img = frame[y:y+h, x:x+w]

                name, confidence = recognize_face(face_img)

                label = name if confidence is None else f"{name} ({confidence}%)"

                cv2.putText(
                    frame,
                    label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

                if name != "Unknown":
                    present_students.add(name)

            cv2.imshow("Captured", frame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()

            return present_students


# =========================
# LIVE MODE
# =========================

def live_attendance(cam, detector):

    print("Live mode started. Press 'q' to quit.")

    detection_count = defaultdict(int)
    present_students = set()

    while True:

        frame = cam.get_frame()

        if frame is None:
            break

        faces = detector.detect_faces(frame)

        for (x, y, w, h) in faces:

            face_img = frame[y:y+h, x:x+w]

            name, confidence = recognize_face(face_img)

            if name != "Unknown":

                detection_count[name] += 1

                if detection_count[name] >= REQUIRED_DETECTIONS:

                    present_students.add(name)

                label = f"{name} ({confidence}%)"

                cv2.putText(
                    frame,
                    label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

        frame = detector.draw_faces(frame, faces)

        cv2.imshow("Live Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    return present_students


# =========================
# MAIN
# =========================

def main():

    load_embeddings()

    cam = Camera()
    detector = FaceDetector()

    if ATTENDANCE_MODE == "snapshot":

        present_students = snapshot_attendance(
            cam,
            detector
        )

    else:

        present_students = live_attendance(
            cam,
            detector
        )

    cam.release()

    write_attendance(present_students)

    print("\nAttendance saved successfully.")


if __name__ == "__main__":
    main()

