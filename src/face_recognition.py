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
