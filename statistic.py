from utils import extract_frames, patching
from multiprocessing import Pool
from collections import Counter
from pathlib import Path
import numpy as np
import os



data_path = Path("data/UCF11")
NUM_WORKERS = int( os.cpu_count() * 0.6 )


data_path_list = [str(video) for video in data_path.rglob("*.mpg")]

pool = Pool(NUM_WORKERS)
dataset = pool.imap(extract_frames, data_path_list)


data = [len(list(video)) for video in dataset]


print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Min:", np.min(data))
print("Max:", np.max(data))


counter = dict(sorted(Counter(data).items()))
print("Total:", np.sum(list(counter.keys())))


sum = 0
total = np.sum(list(counter.values()))
for k, v in counter.items():
    sum += v
    counter[k] = round(sum/total, 2)

print("Num_frame / %:", counter)
