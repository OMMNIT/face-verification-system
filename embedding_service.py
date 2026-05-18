from deepface import DeepFace
import numpy as np
import cv2
import tempfile


def get_embedding(face):
    face = cv2.resize(face, (160, 160))

    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp_file.name, face)

    embedding = DeepFace.represent(
        img_path=temp_file.name,
        model_name="Facenet",
        enforce_detection=False
    )[0]["embedding"]

    return np.array(embedding)
