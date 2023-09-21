from typing import Any, Callable, Optional, Tuple, List
from multiprocessing import Pool
from pathlib import Path
import os
import cv2

import torch
import torchvision.transforms as T
from torchvision.datasets import VisionDataset
from torchvision.io import read_video
from rich.progress import track
import numpy as np



class VideoProcessing():
    def __init__(
        self, 
        sampling_value: int, 
        num_frames: int, 
        size: Tuple[int, int]
    ) -> None:
        self.sampling_value = sampling_value
        self.num_frames = num_frames
        self.size = size


    def __call__(self, path: str) -> List[np.ndarray]:
        return self.auto(path)


    def load(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileExistsError("File not found!")
        return read_video(path, pts_unit="sec")[0].numpy()


    def sampling(self, video: np.ndarray, value: int) -> np.ndarray:
        """
        Pick up 1 frame every n frame (n = value)
        """
        return video[::value, ...]


    def balancing(self, video: np.ndarray, value: int) -> np.ndarray:
        """
        Change number of frame in a video
        - Cut both head and last if > max_frame
        - Append black frame to end if < max_frame
        """
        video_lenth = len(video)
        if video_lenth > value:
            middle_frame = video_lenth // 2
            m = value // 2
            r = value % 2
            video = video[middle_frame - m : middle_frame + m + r]
        elif video_lenth < value:
            zeros_array = np.zeros((value, *video.shape[1:]), dtype=np.uint8)
            zeros_array[:video_lenth, ...] = video
            video = zeros_array
        return video


    def resize(self, video: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Change size of video
        """
        return np.array([cv2.resize(frame, size) for frame in video])


    def auto(self, path: np.ndarray) -> List[np.ndarray]:
        out = self.load(path)
        out = self.sampling(out, self.sampling_value)
        out = self.balancing(out, self.num_frames)
        out = self.resize(out, self.size)
        return out



class UCF11Dataset(VisionDataset):
    def __init__(
            self,
            root: str,
            transforms: Optional[Callable] = None,
            sampling_value: int = 3,
            num_frames: int = 64,
            size: Tuple[int, int] = (224, 224),
            num_wokers: int = 1,
            chunking: int = 100,
        ) -> None:
        super().__init__(root, transforms)
        self.root = root
        self.transforms = transforms
        self.video_setting = {
            "sampling_value": sampling_value,
            "num_frames": num_frames,
            "size": size,
        }
        self.num_wokers = num_wokers
        self.data, self.labels = self._setup(chunking)


    def _divide_list(self, input_list: List[Any], chunk_size: int) -> List[List]:
        """
        Divide list into smaller chunk
        """
        return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


    def _setup(self, chunking: int) -> Tuple[List, List]:        
        path_list = [str(video) for video in Path(self.root).rglob("*.mpg")]
        path_list_chunks = self._divide_list(path_list, chunking)

        VP = VideoProcessing(**self.video_setting)

        dataset = []
        with Pool(self.num_wokers) as pool:
            for chunk in track(path_list_chunks, "Preparing data:"):
                dataset.extend(pool.map(VP, chunk))

        labels = os.listdir(self.root)

        return dataset, labels


    def __len__(self) -> int:
        return len(self.data) if self.data else 0


    def __getitem__(self, index: int) -> Any:
        return self.data[index], self.labels[index]



dataset = UCF11Dataset("data/UCF11", num_wokers=22)

print(np.array(dataset.data))



