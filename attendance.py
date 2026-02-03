import csv
import os
from datetime import datetime

FILE_NAME = "attendance.csv"


# Create CSV file if it does not exist
def create_file_if_not_exists():
    if not os.path.exists(FILE_NAME):
        with open(FILE_NAME, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["StudentName", "Date", "Time", "SessionID"])


# Check if attendance already exists for the session
def already_marked(student_name, session_id):
    if not os.path.exists(FILE_NAME):
        return False

    with open(FILE_NAME, mode="r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if (
                row["StudentName"] == student_name
                and row["SessionID"] == session_id
            ):
                return True

    return False


# Mark attendance
def mark_attendance(student_name, session_id):
    create_file_if_not_exists()

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if already_marked(student_name, session_id):
        print("Attendance already marked.")
        return

    with open(FILE_NAME, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([student_name, date_str, time_str, session_id])

    print("Attendance recorded.")


# Test run when file is executed directly
#if __name__ == "__main__":
 #   mark_attendance("Rahul", "CS101_2026-02-04")