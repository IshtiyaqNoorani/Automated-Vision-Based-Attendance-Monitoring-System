import insightface
import numpy as np
import os
import cv2

MODEL_NAME = "buffalo_l"
DATASET_PATH = "data/registered_faces"

SIMILARITY_THRESHOLD = 0.45

app = None
embeddings_db = {}


def initialize_engine():

    global app
    global embeddings_db

    print("Loading InsightFace model...")

    app = insightface.app.FaceAnalysis(name=MODEL_NAME)
    app.prepare(ctx_id=-1)  # CPU mode

    print("Model loaded.")

    print("Loading registered faces...")

    embeddings_db = {}

    for person in os.listdir(DATASET_PATH):

        person_path = os.path.join(DATASET_PATH, person)

        if not os.path.isdir(person_path):
            continue

        embeddings_db[person] = []

        for img_name in os.listdir(person_path):

            img_path = os.path.join(person_path, img_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            faces = app.get(img)

            if len(faces) == 0:
                continue

            embedding = faces[0].embedding

            embeddings_db[person].append(embedding)

    print("Embeddings loaded.")


def recognize_frame(frame):

    faces = app.get(frame)

    results = []

    for face in faces:

        embedding = face.embedding

        best_match = None
        best_similarity = -1

        for person in embeddings_db:

            for stored_embedding in embeddings_db[person]:

                similarity = np.dot(
                    embedding,
                    stored_embedding
                ) / (
                    np.linalg.norm(embedding) *
                    np.linalg.norm(stored_embedding)
                )

                if similarity > best_similarity:

                    best_similarity = similarity
                    best_match = person

        box = face.bbox.astype(int)

        x, y, x2, y2 = box
        w = x2 - x
        h = y2 - y

        if best_similarity > SIMILARITY_THRESHOLD:

            confidence = best_similarity * 100

            results.append({
                "name": best_match,
                "confidence": round(confidence, 2),
                "box": (x, y, w, h)
            })

        else:

            results.append({
                "name": "Unknown",
                "confidence": None,
                "box": (x, y, w, h)
            })

    return results
