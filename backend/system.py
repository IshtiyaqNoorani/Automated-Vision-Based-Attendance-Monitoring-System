import cv2
import os
import numpy as np
from deepface import DeepFace

from src.attendance import write_attendance


# =========================
# CONFIGURATION
# =========================

MODEL_NAME = "ArcFace"
DATASET_PATH = "data/registered_faces"

FACE_SIZE = (160, 160)
MIN_FACE_SIZE = 80

SIMILARITY_THRESHOLD = 0.40


# =========================
# GLOBAL STATE
# =========================

camera = None
face_detector = None
embeddings_db = {}


# =========================
# INITIALIZATION
# =========================

def initialize_system():
    """
    Call this ONCE when application starts.
    Loads embeddings and initializes camera and detector.
    """

    global camera
    global face_detector
    global embeddings_db

    print("Initializing attendance system...")

    # Initialize camera
    camera = cv2.VideoCapture(0)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize face detector
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    # Load embeddings
    embeddings_db = {}

    print("Loading registered faces...")

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

            except Exception as e:

                print(f"Skipping {img_path}")

    print("System initialized successfully.")


# =========================
# CAMERA FUNCTIONS
# =========================

def get_frame():
    """
    Returns current camera frame.
    PyQt frontend will display this frame.
    """

    global camera

    if camera is None:
        raise Exception("Camera not initialized")

    ret, frame = camera.read()

    if not ret:
        return None

    # Fix mirrored webcam
    frame = cv2.flip(frame, 1)

    return frame


def release_camera():
    """
    Call when application closes.
    """

    global camera

    if camera is not None:

        camera.release()

        camera = None


# =========================
# FACE DETECTION
# =========================

def detect_faces(frame):
    """
    Returns list of face bounding boxes.
    """

    global face_detector

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
    )

    return faces


# =========================
# FACE RECOGNITION
# =========================

def recognize_frame(frame):
    """
    Recognizes all faces in a frame.

    Returns:
        List of dictionaries:
        [
            {
                "name": "2408912_Ishtiyaq",
                "confidence": 72.5,
                "box": (x, y, w, h)
            }
        ]
    """

    global embeddings_db

    faces = detect_faces(frame)

    results = []

    for (x, y, w, h) in faces:

        face_img = frame[y:y+h, x:x+w]

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
            best_similarity = -1

            for person in embeddings_db:

                for stored in embeddings_db[person]:

                    similarity = np.dot(
                        embedding,
                        stored
                    ) / (
                        np.linalg.norm(embedding) *
                        np.linalg.norm(stored)
                    )

                    if similarity > best_similarity:

                        best_similarity = similarity
                        best_match = person

            if best_similarity > SIMILARITY_THRESHOLD:

                confidence = best_similarity * 100

                results.append({

                    "name": best_match,
                    "confidence": round(confidence, 2),
                    "box": (x, y, w, h)

                })

            else:

                results.append({

                    "name": "Unknown",
                    "confidence": None,
                    "box": (x, y, w, h)

                })

        except:

            results.append({

                "name": "Unknown",
                "confidence": None,
                "box": (x, y, w, h)

            })

    return results


# =========================
# ATTENDANCE
# =========================

def mark_attendance(recognized_results):
    """
    Saves attendance to CSV.

    recognized_results = output of recognize_frame()
    """

    present_students = set()

    for result in recognized_results:

        if result["name"] != "Unknown":

            present_students.add(result["name"])

    write_attendance(present_students)

    return list(present_students)
