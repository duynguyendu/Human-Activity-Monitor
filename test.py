import torch
from utils import extract_frames, patching
from pathlib import Path
import cv2
import numpy as np
from multiprocessing import Pool



device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("data/UCF11")

data_path_list = [video for video in data_path.rglob("*.mpg")][:100]


with Pool(22) as p:
    dataset = p.map(extract_frames, data_path_list)


print(np.mean([len(video) for video in dataset]))
