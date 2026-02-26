import cv2
import numpy as np
import insightface
import os


# =========================
# CONFIGURATION
# =========================

DATASET_PATH = "data/registered_faces"

SIMILARITY_THRESHOLD = 0.45

MODEL_NAME = "buffalo_l"
frame_count = 0

# =========================
# LOAD MODEL
# =========================

print("Loading InsightFace model...")

app = insightface.app.FaceAnalysis(name=MODEL_NAME)

app.prepare(ctx_id=-1)  # CPU mode

print("Model loaded successfully.")


# =========================
# LOAD REGISTERED FACES
# =========================

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

print("Registered faces loaded.")


# =========================
# START CAMERA
# =========================

cap = cv2.VideoCapture(0)

# Set high resolution for better accuracy
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("\nCamera started.")
print("Press 'q' to quit.\n")


# =========================
# RECOGNITION LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    display_frame = frame.copy()

    frame_count += 1

    # Only run heavy recognition every 5 frames
    if frame_count % 5 == 0:

        faces = app.get(frame)

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

            x1, y1, x2, y2 = box

            if best_similarity > SIMILARITY_THRESHOLD:

                confidence = best_similarity * 100
                label = f"{best_match} ({confidence:.1f}%)"
                color = (0,255,0)

            else:

                label = "Unknown"
                color = (0,0,255)

            cv2.rectangle(display_frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(display_frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

    cv2.imshow("InsightFace - Optimized", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# CLEANUP
# =========================

cap.release()

cv2.destroyAllWindows()

print("Camera closed.")
