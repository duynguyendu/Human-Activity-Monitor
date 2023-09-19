from typing import List, Tuple
import time
import numpy as np
import torch
import cv2


def patching(image: np.ndarray, patch_size: Tuple[int, int]) -> np.ndarray[np.ndarray]:
    """Divide image into smaller, non-overlapping regions or patches"""

    patches = []
    for y in range(0, image.shape[0], patch_size[1]):
        for x in range(0, image.shape[1], patch_size[0]):
            patch = image[y:y + patch_size[1], x:x + patch_size[0]]
            patches.append(patch)
    return np.array(patches)
