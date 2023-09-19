from collections import Counter
import torch
from utils import extract_frames, patching
from pathlib import Path
import cv2
import numpy as np
from multiprocessing import Pool

import os


device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("data/UCF11")

data_path_list = [str(video) for video in data_path.rglob("*.mpg")]


NUM_WORKERS = 44


pool = Pool(NUM_WORKERS)
dataset = pool.imap(extract_frames, data_path_list)

data = [len(list(video)) for video in dataset]


print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Min:", np.min(data))
print("Max:", np.max(data))

sum = 0
counter = dict(sorted(Counter(data).items()))
print("Total:", np.sum(list(counter.keys())))
total = np.sum(list(counter.values()))

for k, v in counter.items():
    sum += v
    counter[k] = round(sum/total, 2)

print("Counter:", counter)