import cv2

from backend.system import (
    initialize_system,
    get_frame,
    recognize_frame,
    mark_attendance,
    release_camera
)


def main():

    initialize_system()

    print("\nPress 'c' to capture attendance")
    print("Press 'q' to quit\n")

    while True:

        frame = get_frame()

        if frame is None:
            continue

        preview = frame.copy()

        cv2.imshow("Attendance System", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):

            break

        if key == ord('c'):

            results = recognize_frame(frame)

            for result in results:

                x, y, w, h = result["box"]

                name = result["name"]
                confidence = result["confidence"]

                if confidence is not None:

                    label = f"{name} ({confidence}%)"

                else:

                    label = "Unknown"

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

            cv2.imshow("Captured", frame)

            mark_attendance(results)

            print("Attendance saved.")

            cv2.waitKey(0)

            break

    release_camera()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
