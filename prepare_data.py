from utils import extract_frames
from multiprocessing import Pool
from pathlib import Path
from typing import List
import cv2
import os

from rich.progress import track
from PIL import Image
import numpy as np



def cap_frame(video: np.ndarray, max_frame: int) -> np.ndarray:
    """
    Change number of frame in a video
    - Cut both head and last if > max_frame
    - Append black frame to end if < max_frame
    """
    video_lenth = len(video)
    if video_lenth > max_frame:
        middle_frame = video_lenth // 2
        m = max_frame // 2
        r = max_frame % 2
        video = video[middle_frame - m : middle_frame + m + r]
    elif video_lenth < max_frame:
        zeros_array = np.zeros((max_frame, *video.shape[1:]), dtype=np.uint8)
        zeros_array[:video_lenth, ...] = video
        video = zeros_array
    return video


def reshape_video(video: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Change size of video
    """
    x_size, y_size = video.shape[1], video.shape[2]
    if x_size > target_size[0] or y_size > target_size[1]:
        video = video[:, :target_size[0], :target_size[1], :]
    elif x_size < target_size[0] or y_size < target_size[1]:
        zeros_array = np.zeros((video.shape[0], target_size[0], target_size[1], video.shape[-1]), dtype=np.uint8)
        zeros_array[:, :x_size, :y_size, :] = video
        video = zeros_array
    return video


def divide_list(input_list: List, chunk_size: int) -> List[List]:
    """
    Divide list into smaller chunk
    """
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def pipeline(video_path):
    """
    Main pipeline:
    - Preprocess
    - Saving
    """
    SAVE_FOLDER = "processed_dataa"

    MAX_FRAME = 256
    X_SIZE = 240
    Y_SIZE = 320

    video = extract_frames(video_path)

    video = cap_frame(video, MAX_FRAME)

    video = reshape_video(video, (X_SIZE, Y_SIZE))

    splitted_path = video_path.split("/")
    folder_path = "/".join(splitted_path[1:-1])
    filename = splitted_path[-1].split(".")[0]

    SAVE_PATH = f"{SAVE_FOLDER}/{folder_path}/{filename}"
    os.makedirs(SAVE_PATH, exist_ok=True)

    for j, frame in enumerate(video):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.save(f"{SAVE_PATH}/{j}.jpg")



if __name__=="__main__":
    """
    Main process
    """
    DATA_LIMIT = None
    DATA_PATH = "data/UCF11"
    NUM_WORKERS = int( os.cpu_count() * 0.6 )

    data_path_list = [str(video) for video in Path(DATA_PATH).rglob("*.mpg")][:DATA_LIMIT]

    pool = Pool(NUM_WORKERS)
    data_pool = pool.imap(pipeline, data_path_list)

    [_ for _, _ in zip(data_pool, track(data_path_list))]
