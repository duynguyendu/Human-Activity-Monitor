from typing import List, Tuple
import numpy as np
import cv2



def extract_frames(path: str) -> np.ndarray[np.ndarray]:
    """
    Create list of frames from video
    """
    if not isinstance(path, str):
        raise TypeError("Path must be string.")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError("Could not open video file.")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)


def patching(image: np.ndarray, patch_size: Tuple[int, int]) -> np.ndarray[np.ndarray]:
    """Divide image into smaller, non-overlapping regions or patches"""

    patches = []
    for y in range(0, image.shape[0], patch_size[1]):
        for x in range(0, image.shape[1], patch_size[0]):
            patch = image[y:y + patch_size[1], x:x + patch_size[0]]
            patches.append(patch)
    return np.array(patches)
