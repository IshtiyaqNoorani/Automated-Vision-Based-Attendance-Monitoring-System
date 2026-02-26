import cv2
from backend.system import initialize_system, get_frame, recognize_frame, release_camera

initialize_system()

print("Press 'q' to quit")

while True:

    frame = get_frame()

    if frame is None:
        continue

    results = recognize_frame(frame)

    for result in results:

        x, y, w, h = result["box"]
        name = result["name"]
        confidence = result["confidence"]

        if confidence is not None:
            label = f"{name} ({confidence:.1f}%)"
        else:
            label = "Unknown"

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Backend Test - Live Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

release_camera()
cv2.destroyAllWindows()
