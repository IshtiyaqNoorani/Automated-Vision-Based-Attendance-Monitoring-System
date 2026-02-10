import cv2
from collections import defaultdict

from src.camera import Camera
from src.face_detection import FaceDetector
from src.face_recognition import (
    load_registered_embeddings,
    recognize_face
)
from src.attendance import write_attendance


def main():
    cam = Camera()
    detector = FaceDetector()

    # ---- Accuracy-first parameters ----
    REQUIRED_DETECTIONS = 8       # multiple confirmations
    MIN_CONFIDENCE = 80           # strong match only
    FACE_SIZE = (160, 160)        # FaceNet input size

    # Track detections
    present_students = set()
    detection_count = defaultdict(int)

    print("Loading registered face embeddings...")
    embeddings_db = load_registered_embeddings()
    print("Embeddings loaded successfully.")
    print("System running. Press 'q' to end session.")

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        faces = detector.detect_faces(frame)

        for (x, y, w, h) in faces:
            # Crop face
            face_img = frame[y:y+h, x:x+w]

            # Skip very small faces (too far / unreliable)
            if face_img.shape[0] < 60 or face_img.shape[1] < 60:
                continue

            # Resize for FaceNet (important for distance)
            face_img = cv2.resize(face_img, FACE_SIZE)

            name, confidence = recognize_face(face_img, embeddings_db)

            # Strong confirmation logic
            if (
                name != "Unknown"
                and confidence is not None
                and confidence >= MIN_CONFIDENCE
            ):
                detection_count[name] += 1

                if detection_count[name] >= REQUIRED_DETECTIONS:
                    present_students.add(name)

            # Display label
            label = "Unknown"
            if confidence is not None:
                label = f"{name} ({confidence}%)"

            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        frame = detector.draw_faces(frame, faces)
        cv2.imshow("Automated Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()

    print("Writing attendance...")
    write_attendance(present_students)
    print("Attendance saved successfully.")


if __name__ == "__main__":
    main()

