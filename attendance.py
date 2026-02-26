import csv
import os
from datetime import datetime

FILE_NAME = "attendance.csv"


def create_file_if_not_exists():
    if not os.path.exists(FILE_NAME):
        with open(FILE_NAME, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["StudentName", "Date", "Time", "SessionID"])


def generate_session_id():
    now = datetime.now()
    return now.strftime("SESSION_%Y%m%d_%H%M")


def already_marked(student_name, session_id):
    if not os.path.exists(FILE_NAME):
        return False

    with open(FILE_NAME, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["StudentName"] == student_name and row["SessionID"] == session_id:
                return True

    return False


def mark_attendance(student_name, session_id):
    create_file_if_not_exists()

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if already_marked(student_name, session_id):
        return

    with open(FILE_NAME, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([student_name, date_str, time_str, session_id])

    print("Recorded:", student_name)


def write_attendance(student_set):
    session_id = generate_session_id()

    for student in student_set:
        mark_attendance(student, session_id)
