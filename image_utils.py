import cv2
import numpy as np
from fastapi import UploadFile


def read_image(file: UploadFile):
    contents = file.file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img