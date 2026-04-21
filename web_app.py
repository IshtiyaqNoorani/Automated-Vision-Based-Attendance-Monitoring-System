from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import os
import csv
from datetime import datetime
from backend.engine import Engine

app = Flask(__name__)
engine = Engine()

# 🔹 Session state
current_session = set()
session_active = False


@app.route("/")
def index():
    return render_template("index.html")


# ▶️ START SESSION
@app.route("/start", methods=["POST"])
def start_session():
    global current_session, session_active
    current_session = set()
    session_active = True
    return jsonify({"status": "started"})


# ⏹ END SESSION
@app.route("/end", methods=["POST"])
def end_session():
    global session_active
    session_active = False

    save_attendance(current_session)

    return jsonify({
        "status": "ended",
        "count": len(current_session),
        "students": list(current_session)
    })


# 🎥 PROCESS FRAME
@app.route("/process", methods=["POST"])
def process():
    global current_session, session_active

    file = request.files.get("frame")
    if not file:
        return jsonify([])

    img_bytes = file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = []

    try:
        faces = engine.app.get(frame)
    except Exception as e:
        print("Error:", e)
        return jsonify([])

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        name, score = engine.match(face.embedding)

        if session_active and name != "Unknown":
            current_session.add(name)

        results.append({
            "name": name,
            "score": float(score),
            "box": [x1, y1, x2, y2]
        })

    return jsonify(results)


# 💾 SAVE ATTENDANCE
def save_attendance(names):
    if not names:
        return

    file_exists = os.path.exists("attendance.csv")

    with open("attendance.csv", "a") as f:
        if not file_exists:
            f.write("Name,Time,Status\n")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for name in names:
            f.write(f"{name},{now},Present\n")


# 📊 ANALYTICS
@app.route("/analytics")
def analytics():
    registered_path = "data/registered_faces"
    attendance_file = "attendance.csv"

    total_registered = len(os.listdir(registered_path)) if os.path.exists(registered_path) else 0

    present = set()

    if os.path.exists(attendance_file):
        with open(attendance_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                present.add(row["Name"])

    present_count = len(present)
    absent_count = max(total_registered - present_count, 0)

    rate = (present_count / total_registered * 100) if total_registered > 0 else 0

    return jsonify({
        "total": total_registered,
        "present": present_count,
        "absent": absent_count,
        "rate": round(rate, 2)
    })


# 📥 DOWNLOAD CSV
@app.route("/download")
def download():
    if not os.path.exists("attendance.csv"):
        return "No attendance file found", 404

    return send_file("attendance.csv", as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)