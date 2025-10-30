import cv2
import numpy as np
from ip_config import TACTILE_CAMERA

for side in TACTILE_CAMERA:
    cap = cv2.VideoCapture(TACTILE_CAMERA[side])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise ValueError(f"Cannot open {side} tactile camera: {TACTILE_CAMERA[side]}")
    
    _, ref = cap.read()
    input(f"Please press on the {side} tactile sensor, and then press enter")
    for _ in range(10):
        _, img = cap.read()

    compare = np.hstack((ref, img))
    cv2.imwrite(f"data/{side}.png", compare)

    diff = np.max(ref.astype(float) - img.astype(float))

    if diff > 20:
        print(f"{side} tactile sensor is OK (diff: {diff})")
    else:
        raise ValueError(f"{side} tactile sensor is not responding properly (diff: {diff})")

    cap.release()

