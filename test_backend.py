from backend.system import initialize_system, get_frame, recognize_frame, mark_attendance, release_camera

initialize_system()

frame = get_frame()

results = recognize_frame(frame)

print("Recognition results:")
print(results)

present = mark_attendance(results)

print("Attendance marked:", present)

release_camera()
