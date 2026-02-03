"""
Face Recognition Module
-----------------------
Responsibilities:
- Load registered student face images
- Extract facial embeddings using DeepFace
- Match incoming face embeddings with known students

Does NOT:
- Access camera
- Detect faces
- Mark attendance
"""

import os
import numpy as np
from deepface import DeepFace

REGISTERED_FACES_DIR = "data/registered_faces"


def load_registered_embeddings():
    """
    Loads all registered student faces and extracts embeddings.

    Returns:
        dict: {
            "ROLLNO_NAME": [embedding1, embedding2, ...]
        }
    """
    embeddings_db = {}

    if not os.path.exists(REGISTERED_FACES_DIR):
        raise FileNotFoundError("registered_faces directory not found")

    for student in os.listdir(REGISTERED_FACES_DIR):
        student_path = os.path.join(REGISTERED_FACES_DIR, student)

        if not os.path.isdir(student_path):
            continue

        embeddings_db[student] = []

        for img in os.listdir(student_path):
            img_path = os.path.join(student_path, img)

            try:
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name="Facenet",
                    enforce_detection=False
                )[0]["embedding"]

                embeddings_db[student].append(np.array(embedding))

            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    return embeddings_db

def cosine_distance(vec1, vec2):
    """
    Computes cosine distance between two vectors.
    Smaller value = more similar faces.
    """
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return 1.0
    return 1 - np.dot(vec1, vec2) / denom


def recognize_face(face_img, embeddings_db, threshold=0.4):
    """
    Matches a given face image against known embeddings.

    Args:
        face_img (numpy array): Cropped face image from detection module
        embeddings_db (dict): Output of load_registered_embeddings()
        threshold (float): Distance threshold for recognition

    Returns:
        (name, confidence)
        name = student folder name OR "Unknown"
        confidence = percentage match OR None
    """
    try:
        result = DeepFace.represent(
            img_path=face_img,
            model_name="Facenet",
            enforce_detection=False
        )
        live_embedding = np.array(result[0]["embedding"])
    except Exception:
        return "Unknown", None

    best_match = None
    best_distance = float("inf")

    for student, embeddings in embeddings_db.items():
        for stored_embedding in embeddings:
            distance = cosine_distance(live_embedding, stored_embedding)

            if distance < best_distance:
                best_distance = distance
                best_match = student

    if best_distance < threshold:
        confidence = round((1 - best_distance) * 100, 2)
        return best_match, confidence

    return "Unknown", None

