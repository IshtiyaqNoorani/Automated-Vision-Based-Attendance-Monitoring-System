import os
from datetime import datetime

FILE_NAME = "attendance.csv"


def get_all_students(registered_faces_dir="data/registered_faces"):
    students = []
    for name in os.listdir(registered_faces_dir):
        path = os.path.join(registered_faces_dir, name)
        if os.path.isdir(path):
            students.append(name)
    return students


def write_attendance(present_students):
    time_str = datetime.now().strftime("%H:%M:%S")
    all_students = get_all_students()

    with open(FILE_NAME, "w") as file:
        # Header
        file.write(f"{'Name':<20} {'Time':<10} {'Status'}\n")
        file.write("-" * 40 + "\n")

        # Rows
        for student in all_students:
            status = "Present" if student in present_students else "Absent"
            file.write(f"{student:<20} {time_str:<10} {status}\n")

