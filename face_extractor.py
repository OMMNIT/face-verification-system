from retinaface import RetinaFace
import cv2


def extract_face_from_document(img):
    detections = RetinaFace.detect_faces(img)

    if detections is None or len(detections) == 0:
        return None, "No face detected"

    best_face = None
    max_area = 0

    for key in detections:
        face = detections[key]
        x1, y1, x2, y2 = face["facial_area"]

        area = (x2 - x1) * (y2 - y1)

        if area > max_area:
            max_area = area
            best_face = (x1, y1, x2, y2)

    x1, y1, x2, y2 = best_face

    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face_img = img[y1:y2, x1:x2]

    if face_img.size == 0:
        return None, "Invalid face crop"

    return face_img, None
