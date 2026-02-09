import cv2

from src.camera import Camera
from src.face_detection import FaceDetector
from src.face_recognition import (
    load_registered_embeddings,
    recognize_face
)
from src.attendance import AttendanceManager


def main():
    cam = Camera()
    detector = FaceDetector()
    attendance = AttendanceManager()

    print("Loading registered face embeddings...")
    embeddings_db = load_registered_embeddings()
    print("Embeddings loaded successfully.")

    print("Starting Automated Attendance System (press 'q' to quit)")

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        faces = detector.detect_faces(frame)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            name, confidence = recognize_face(face_img, embeddings_db)

            if name != "Unknown":
                attendance.mark_attendance(name)

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

