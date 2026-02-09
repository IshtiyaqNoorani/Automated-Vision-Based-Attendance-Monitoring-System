import cv2
from datetime import datetime

from src.camera import Camera
from src.face_detection import FaceDetector
from src.face_recognition import (
    load_registered_embeddings,
    recognize_face
)
from src.attendance import mark_attendance


def main():
    cam = Camera()
    detector = FaceDetector()

    # Create a unique session ID (per class run)
    session_id = datetime.now().strftime("CLASS_%Y-%m-%d_%H-%M")

    print("Loading registered face embeddings...")
    embeddings_db = load_registered_embeddings()
    print("Embeddings loaded successfully.")

    print("Starting Automated Attendance System")
    print(f"Session ID: {session_id}")
    print("Press 'q' to quit")

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        faces = detector.detect_faces(frame)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            name, confidence = recognize_face(face_img, embeddings_db)

            if name != "Unknown":
                mark_attendance(name, session_id)

            label = name
            if confidence is not None:
                label = f"{name} ({confidence}%)"

            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        frame = detector.draw_faces(frame, faces)
        cv2.imshow("Automated Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()


if __name__ == "__main__":
    main()

