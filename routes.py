from fastapi import APIRouter, UploadFile, File
from face_extractor import extract_face_from_document
from embedding_service import get_embedding
from matcher import verify_match
from image_utils import read_image

router = APIRouter()


@router.post("/verify-face")
async def verify_face(
    kyc: UploadFile = File(...),
    live: UploadFile = File(...),
    threshold: float = 0.6
):

    kyc_img = read_image(kyc)
    live_img = read_image(live)

    kyc_face, err = extract_face_from_document(kyc_img)

    if err:
        return {"error": err}

    emb1 = get_embedding(kyc_face)
    emb2 = get_embedding(live_img)

    # pass threshold here
    decision, score = verify_match(
        emb1,
        emb2,
        threshold
    )

    return {
        "decision": decision,
        "similarity": float(score),
        "threshold": threshold
    }