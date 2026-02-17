import cv2
from collections import defaultdict
from deepface import DeepFace
import numpy as np

from attendance import write_attendance


# =========================
# CONFIGURATION
# =========================

ATTENDANCE_MODE = "snapshot"   # snapshot or live
MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"

MIN_CONFIDENCE = 88
REQUIRED_DETECTIONS = 6

FACE_SIZE = (160, 160)
MIN_FACE_SIZE = 80

DATASET_PATH = "dataset"


# =========================
# CAMERA CLASS
# =========================

class Camera:
    def __init__(self, cam_id=0):
        self.cap = cv2.VideoCapture(cam_id)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()


# =========================
# FACE DETECTOR
# =========================

class FaceDetector:
    def detect_faces(self, frame):
        detections = DeepFace.extract_faces(
            img_path=frame,
            target_size=(224, 224),
            detector_backend=DETECTOR,
            enforce_detection=False,
            align=True
        )

        faces = []

        for d in detections:
            region = d["facial_area"]
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]

            if w > MIN_FACE_SIZE and h > MIN_FACE_SIZE:
                faces.append((x, y, w, h))

        return faces

    def draw_faces(self, frame, faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        return frame


# =========================
# FACE PREPROCESSING
# =========================

def preprocess_face(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    face = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return face


# =========================
# FACE RECOGNITION
# =========================

def recognize_face(face_img):
    try:
        face_img = preprocess_face(face_img)

        dfs = DeepFace.find(
            img_path=face_img,
            db_path=DATASET_PATH,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=False,
            silent=True
        )

        if len(dfs) == 0 or dfs[0].empty:
            return "Unknown", None

        best_match = dfs[0].iloc[0]
        identity = best_match["identity"]
        distance = best_match["distance"]

        name = identity.split("/")[-2]

        confidence = max(0, 100 - distance * 100)

        if confidence < MIN_CONFIDENCE:
            return "Unknown", confidence

        return name, round(confidence,2)

    except Exception:
        return "Unknown", None


# =========================
# SNAPSHOT MODE
# =========================

def snapshot_attendance(cam, detector):
    print("Stabilizing camera...")

    for _ in range(10):
        cam.get_frame()

    present_students = set()

    # Capture multiple frames
    for _ in range(6):
        frame = cam.get_frame()
        if frame is None:
            continue

        faces = detector.detect_faces(frame)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, FACE_SIZE, interpolation=cv2.INTER_CUBIC)

            name, confidence = recognize_face(face_img)

            if name != "Unknown":
                present_students.add(name)

    return present_students


# =========================
# LIVE MODE
# =========================

def live_attendance(cam, detector):
    print("Press Q to stop.")

    detection_count = defaultdict(int)
    present_students = set()

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        faces = detector.detect_faces(frame)
        faces = faces[:15]

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, FACE_SIZE, interpolation=cv2.INTER_CUBIC)

            name, confidence = recognize_face(face_img)

            if name != "Unknown":
                detection_count[name] += 1

                if detection_count[name] >= REQUIRED_DETECTIONS:
                    present_students.add(name)

                label = f"{name} ({confidence}%)"
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,255,0), 2)

        frame = detector.draw_faces(frame, faces)
        cv2.imshow("Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    return present_students


# =========================
# MAIN
# =========================

def main():
    cam = Camera()
    detector = FaceDetector()

    if ATTENDANCE_MODE == "snapshot":
        present_students = snapshot_attendance(cam, detector)
    else:
        present_students = live_attendance(cam, detector)

    cam.release()

    write_attendance(present_students)
    print("Attendance saved.")


if __name__ == "__main__":
    main()
