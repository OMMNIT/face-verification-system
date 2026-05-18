# Face Verification System for VBWAC KYC

## Project Overview

This project implements a deep learning–based face verification pipeline for VBWAC (Video-Based Welcome and Authentication Call) KYC verification.

The system compares:
- A face extracted from a KYC document (Aadhaar, Driving License, Passport, etc.)
- A live/selfie image captured during verification

The goal is to determine whether both images belong to the same person.

---

# 1. Problem Statement

In VBWAC processes, customer identity verification is performed using live video verification.

Challenges:
- KYC document images are often blurry and low quality
- Live selfie images are high quality
- User appearance may change over time
  - Beard
  - Hairstyle
  - Aging
  - Lighting
- Need for privacy-safe local inference
- Need for configurable threshold-based verification

The system should:
1. Detect face from KYC document
2. Detect face from live image
3. Generate facial embeddings
4. Compare embeddings using similarity metrics
5. Return MATCH / REVIEW / NO_MATCH decision

---

# 2. System Architecture

```text
KYC Image + Live Image
        ↓
Image Decoding
        ↓
Face Detection (RetinaFace)
        ↓
Face Embedding Generation (FaceNet)
        ↓
Cosine Similarity
        ↓
Decision Engine
        ↓
MATCH / REVIEW / NO_MATCH
```

---

# 3. Technologies Used

| Component | Technology |
|---|---|
| Backend API | FastAPI |
| Face Detection | RetinaFace |
| Face Embedding | FaceNet |
| Wrapper Library | DeepFace |
| Image Processing | OpenCV |
| Similarity Metric | Cosine Similarity |
| Programming Language | Python |

---

# 4. Project Structure

```text
FVS/
│
├── main.py
├── routes.py
├── embedding_service.py
├── matcher.py
├── image_utils.py
├── face_extractor.py
├── requirements.txt
```

---

# 5. Workflow Explanation

## Step 1: Upload Images

User uploads:
- KYC document image
- Live/selfie image

---

## Step 2: Read Images

Images are converted into numpy arrays using OpenCV.

```python
contents = file.file.read()
np_arr = np.frombuffer(contents, np.uint8)
img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
```

---

## Step 3: Face Detection

The system uses RetinaFace to detect and crop faces.

### Why RetinaFace?
- Works well on low-quality images
- Accurate face localization
- Good for KYC documents

Reference:
https://arxiv.org/abs/1905.00641

---

## Step 4: Face Embedding Generation

The cropped face is converted into a numerical vector using FaceNet.

### What is FaceNet?
FaceNet converts a face image into a high-dimensional embedding vector.

Example:

```text
Face → [0.12, -0.45, 0.91, ...]
```

These embeddings represent identity-specific facial features.

Reference:
https://arxiv.org/abs/1503.03832

---

## Step 5: Similarity Calculation

Cosine similarity is used to compare embeddings.

Formula:

```text
similarity = (A · B) / (||A|| × ||B||)
```

Reference:
https://en.wikipedia.org/wiki/Cosine_similarity

---

## Step 6: Decision Logic

```python
if score >= threshold:
    MATCH
elif score >= threshold - margin:
    REVIEW
else:
    NO_MATCH
```

---

# 6. Current Model Details

Current implementation uses:

- Model: FaceNet
- Library: DeepFace
- Detector: RetinaFace

Current pipeline:

```text
Image → RetinaFace → FaceNet → Cosine Similarity
```

---

# 7. Why FaceNet Was Chosen

Advantages:
- Easy to integrate
- Lightweight
- Good baseline performance
- Widely used face recognition model

Limitations:
- Lower accuracy on KYC vs selfie comparison
- Sensitive to image quality differences
- Not ideal for severe pose/lighting variation

---

# 8. Challenges Faced

## 8.1 KYC Image Quality

KYC document images are:
- Compressed
- Blurry
- Printed/scanned
- Low resolution

This reduces embedding quality.

---

## 8.2 Domain Mismatch

KYC image and live image come from different domains.

| KYC Image | Live Image |
|---|---|
| Printed | Real-time camera |
| Low quality | High quality |
| Old photo | Current appearance |

This naturally reduces similarity score.

---

## 8.3 Appearance Changes

Users may change appearance:
- Beard
- Hairstyle
- Aging
- Weight changes
- Lighting differences

---

# 9. Observed Results

| Scenario | Similarity Range |
|---|---|
| Same person (good quality) | 0.7 – 0.9 |
| Same person (KYC vs selfie) | 0.4 – 0.7 |
| Different person | < 0.4 |

Observation:
Even production systems do not achieve extremely high similarity for KYC vs selfie comparisons.

---

# 10. Why Similarity Is Not Always High

Face verification is based on:
- Structural facial geometry
- Deep embeddings

It is NOT based on:
- Exact image matching
- Pixel comparison
- Permanent marks

Therefore:
- Low-quality KYC images produce lower similarity scores
- Appearance changes reduce score further

---

# 11. Why Not Use Retina Scan or Permanent Marks

## Retina Scan

Not feasible because:
- Requires infrared hardware
- Requires dedicated biometric scanner
- Cannot be done using standard camera images

Reference:
https://en.wikipedia.org/wiki/Retinal_scan

---

## Permanent Marks

Not reliable because:
- Not always visible
- Lighting dependent
- Can be hidden or altered
- Not robust for automation

Industry systems typically avoid this approach.

---

# 12. Why Gemini or LLM Models Are Not Used

Gemini and other LLMs are:
- Generative AI models
- Non-deterministic
- Not designed for biometric verification

Face verification requires:
- Deterministic embeddings
- Consistent numerical representations

Therefore, specialized face recognition models are preferred.

Reference:
https://deepmind.google/technologies/gemini/

---

# 13. Privacy and Local Processing

The system works locally.

Advantages:
- No external API dependency
- No data leaves system
- Better privacy
- No API cost
- Offline capability after model download

Current implementation:

```text
Local Image → Local Model → Local Decision
```

---

# 14. Current Limitations

## 14.1 No Liveness Detection

System cannot currently detect:
- Printed photo spoofing
- Replay attacks
- Video spoofing

---

## 14.2 Single Image Verification

Currently only one live image is used.

Production systems often use:
- Multiple frames
- Video streams
- Temporal verification

---

## 14.3 Baseline Model

Current system uses FaceNet.

Better models exist:
- ArcFace
- InsightFace

---

# 15. Proposed Future Improvements

## 15.1 ArcFace Upgrade

ArcFace provides:
- Better embedding separation
- Better KYC performance
- Better robustness to variation

Reference:
https://arxiv.org/abs/1801.07698

---

## 15.2 Face Alignment

Aligning eyes and face orientation before embedding generation improves consistency.

---

## 15.3 Liveness Detection

Possible checks:
- Blink detection
- Head movement
- Multi-frame analysis

---

## 15.4 Multi-frame Verification

Instead of single image:
- Capture multiple frames
- Average embeddings
- Improve confidence

---

## 15.5 Image Enhancement

Possible preprocessing:
- CLAHE
- Sharpening
- Denoising
- Contrast enhancement

---

# 16. Sample Decision Strategy

```python
if score >= 0.55:
    return "MATCH"
elif score >= 0.45:
    return "REVIEW"
else:
    return "NO_MATCH"
```

This provides:
- Automatic approval
- Manual review zone
- Automatic rejection

---

# 17. References

## FaceNet Paper
https://arxiv.org/abs/1503.03832

## RetinaFace Paper
https://arxiv.org/abs/1905.00641

## ArcFace Paper
https://arxiv.org/abs/1801.07698

## DeepFace Library
https://github.com/serengil/deepface

## RetinaFace GitHub
https://github.com/serengil/retinaface

## Cosine Similarity
https://en.wikipedia.org/wiki/Cosine_similarity

## Retinal Scan
https://en.wikipedia.org/wiki/Retinal_scan

## Gemini AI
https://deepmind.google/technologies/gemini/

## Deep Face Recognition Survey
https://arxiv.org/abs/1804.06655

---

# 18. Final Conclusion

The project successfully demonstrates a deep learning–based face verification system suitable for VBWAC KYC scenarios.

Current implementation:
- Uses FaceNet embeddings
- Uses RetinaFace detection
- Uses cosine similarity for verification
- Runs completely locally

The system demonstrates:
- Modular architecture
- Threshold-based verification
- Practical KYC workflow understanding

Future improvements include:
- ArcFace integration
- Liveness detection
- Multi-frame verification
- Production-grade optimization

---

# 19. Final Summary

This project implements a modular face verification pipeline capable of comparing KYC document images with live images using deep learning embeddings and cosine similarity.

The system is designed as a baseline prototype with a clear path toward production-level enhancements.

