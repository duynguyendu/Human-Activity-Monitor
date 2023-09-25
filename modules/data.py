from typing import Any, List, Tuple, Union
from multiprocessing import Pool
from pathlib import Path
import shutil
import os

import torch
import torchvision.transforms as T
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import cv2



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
        video = cv2.VideoCapture(path)
        if not video.isOpened():
            raise RuntimeError("Could not open video file.")
        output = [frame for _, frame in iter(video.read, (False, None))]
        video.release()
        return output


    def sampling(self, video: np.ndarray, value: int) -> np.ndarray:
        """
        Pick up 1 frame every n frame (n = value)
        """
        return video[::value] if value else video


    def balancing(self, video: np.ndarray, value: int) -> np.ndarray:
        """
        Change number of frame in a video
        - Cut both head and last if > max_frame
        - Append black frame to end if < max_frame
        """
        if value != 0:
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



class UCF11DataModule(LightningDataModule):
    def __init__(
            self, 
            data_path: str, 
            data_limit: Union[int, float] = None,
            remake_data: bool = False,
            batch_size: int = 32,
            train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
            sampling_value: int = 6,
            num_frames: int = 32,
            image_size: Tuple[int, int] = (224, 224),
            chunk_size: int = 100,
            num_workers: int = 0,
            pin_memory: bool = True,
        ):
        super().__init__()
        self.root = data_path
        self.limit = data_limit
        self.x_path = data_path + "_x"
        self.remake = remake_data
        self.split_size = train_val_test_split
        self.chunk_size = chunk_size
        self.workers = num_workers
        self.video_setting = {
            "sampling_value": sampling_value,
            "num_frames": num_frames,
            "size": image_size,
        }
        self.dl_conf = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        self.transform = T.Compose([
            T.Resize(image_size, antialias=True),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])


    def _limit(self, data: List, limit_value: Union[int, float]=None) -> List:
        """
        Limit data
        """
        if not limit_value:
            return data
        if limit_value > len(data):
            raise ValueError(
                "The limit value must be smaller than the list length "
                "or between 0 and 1 if it is a float."
            )
        if 0 < limit_value < 1:
            limit_value = int(len(data)*limit_value)
        return data[:limit_value]


    def _chunking(self, input_list: List[Any], chunk_size: int) -> List[List]:
        """
        Divide list into smaller chunk
        """
        return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


    def _processing(self, path: str) -> None:
        """
        process and save processed video
        """
        splitted_path = path.split("/")
        file_path = "/".join(splitted_path[2:-1])
        file_name = splitted_path[-1].split(".")[0]
        save_path = f"{self.x_path}/{file_path}/{file_name}"

        os.makedirs(save_path, exist_ok=True)

        VP = VideoProcessing(**self.video_setting)

        video = VP(path)

        for j, frame in enumerate(video):
            img = Image.fromarray(frame)
            img.save(f"{save_path}/{j}.jpg")


    def _load(self, path: List[str]):
        """
        Load data from list of path
        """
        data = self.transform(Image.open(path))
        label = torch.as_tensor(
            self.classes.index(
                path.replace(self.x_path, "").split("/")[1]
            )
        )
        return data, label


    def prepare_data(self):
        """
        Preprocess data
        """
        if not os.path.exists("data/UCF11"):
            raise FileNotFoundError("Dataset not found! ('data/UCF11')")

        save_folder = self.root + "_x"

        if os.path.exists(save_folder):
            if self.remake:
                shutil.rmtree(save_folder)
            else:
                print("Prepared data existed. Skiping...")
                return

        print("Processing data...", end="\r")
        os.makedirs(save_folder, exist_ok=True)

        path_list = [str(video) for video in Path(self.root).rglob(f'*.mpg')]
        path_list = self._limit(path_list, self.limit)

        with Pool(self.workers) as pool:
            pool.map(self._processing, path_list)
        print("Processing data. Done")


    def setup(self, stage: str):
        """
        Setup data
        """
        if not hasattr(self, "data_train"):
            self.dataset = ImageFolder(self.x_path, self.transform)
            self.data_train, self.data_val, self.data_test = random_split(self.dataset, lengths=self.split_size)


    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, **self.dl_conf, shuffle=True)


    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, **self.dl_conf, shuffle=False)


    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, **self.dl_conf, shuffle=False)
