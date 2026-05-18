import numpy as np


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def verify_match(emb1, emb2, threshold):
    score = cosine_similarity(emb1, emb2)

    if score >= threshold:
        return "MATCH", score
    elif score >= threshold - 0.05:
        return "REVIEW", score
    else:
        return "NO_MATCH", score
