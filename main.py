import cv2
from src.camera import Camera
from src.face_detection import FaceDetector
from src.face_recognition import FaceRecognizer
from src.attendance import AttendanceManager

def main():
    cam = Camera()
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    attendance = AttendanceManager()

    print("Starting Automated Attendance System... Press 'q' to exit.")

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        faces = detector.detect_faces(frame)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            name = recognizer.recognize(face_img)

            if name != "Unknown":
                attendance.mark_attendance(name)

            cv2.putText(
                frame,
                name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        frame = detector.draw_faces(frame, faces)
        cv2.imshow("Automated Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()

if __name__ == "__main__":
    main()

