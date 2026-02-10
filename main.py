import cv2
from collections import defaultdict

from src.camera import Camera
from src.face_detection import FaceDetector
from src.face_recognition import (
    load_registered_embeddings,
    recognize_face
)
from src.attendance import write_attendance


# ================================
# CONFIGURATION
# ================================

ATTENDANCE_MODE = "snapshot"   # "snapshot" or "live"

MIN_CONFIDENCE = 80            # percent
REQUIRED_DETECTIONS = 8        # for live mode
FACE_SIZE = (160, 160)         # FaceNet input size
MIN_FACE_SIZE = 60             # ignore very small distant faces


# ================================
# SNAPSHOT MODE (Accuracy-first)
# ================================

def snapshot_attendance(cam, detector, embeddings_db):
    """
    Capture a single high-quality frame and perform attendance.
    Best suited for distant classroom cameras.
    """
    print("Snapshot mode selected.")
    print("Stabilizing camera...")

    # Warm-up frames for auto-focus / exposure
    for _ in range(10):
        cam.get_frame()

    frame = cam.get_frame()
    if frame is None:
        print("Failed to capture snapshot.")
        return set()

    present_students = set()

    faces = detector.detect_faces(frame)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        if face_img.shape[0] < MIN_FACE_SIZE or face_img.shape[1] < MIN_FACE_SIZE:
            continue

        face_img = cv2.resize(face_img, FACE_SIZE)

        name, confidence = recognize_face(face_img, embeddings_db)

        if (
            name != "Unknown"
            and confidence is not None
            and confidence >= MIN_CONFIDENCE
        ):
            present_students.add(name)

        label = name if confidence is None else f"{name} ({confidence}%)"
        cv2.putText(
            frame, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

    frame = detector.draw_faces(frame, faces)
    cv2.imshow("Snapshot Attendance", frame)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    return present_students


# ================================
# LIVE MODE (Demonstration mode)
# ================================

def live_attendance(cam, detector, embeddings_db):
    """
    Continuous live attendance with multi-frame confirmation.
    Shows real-time capability but less stable than snapshot mode.
    """
    print("Live mode selected.")
    print("Press 'q' to end session.")

    detection_count = defaultdict(int)
    present_students = set()

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        faces = detector.detect_faces(frame)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            if face_img.shape[0] < MIN_FACE_SIZE or face_img.shape[1] < MIN_FACE_SIZE:
                continue

            face_img = cv2.resize(face_img, FACE_SIZE)

            name, confidence = recognize_face(face_img, embeddings_db)

            if (
                name != "Unknown"
                and confidence is not None
                and confidence >= MIN_CONFIDENCE
            ):
                detection_count[name] += 1

                if detection_count[name] >= REQUIRED_DETECTIONS:
                    present_students.add(name)

            label = name if confidence is None else f"{name} ({confidence}%)"
            cv2.putText(
                frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        frame = detector.draw_faces(frame, faces)
        cv2.imshow("Live Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    return present_students


# ================================
# MAIN
# ================================

def main():
    cam = Camera()
    detector = FaceDetector()

    print("Loading registered face embeddings...")
    embeddings_db = load_registered_embeddings()
    print("Embeddings loaded successfully.")

    if ATTENDANCE_MODE == "snapshot":
        present_students = snapshot_attendance(cam, detector, embeddings_db)

    elif ATTENDANCE_MODE == "live":
        present_students = live_attendance(cam, detector, embeddings_db)

    else:
        raise ValueError("Invalid ATTENDANCE_MODE")

    cam.release()

    print("Writing attendance...")
    write_attendance(present_students)
    print("Attendance saved successfully.")


if __name__ == "__main__":
    main()

