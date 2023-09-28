from collections import Counter
from typing import List, Tuple, Union
from multiprocessing import Pool
from pathlib import Path
from rich import print
import glob
import os

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

from lightning.pytorch import LightningDataModule

from sklearn.model_selection import train_test_split
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



class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label



class UCF11DataModule(LightningDataModule):
    def __init__(
            self, 
            data_path: str, 
            batch_size: int = 32,
            train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
            sampling_value: int = 0,
            num_frames: int = 0,
            image_size: Tuple[int, int] = (224, 224),
            num_workers: int = 0,
            pin_memory: bool = True,
        ):
        super().__init__()
        self.root = data_path
        self.x_path = data_path + "_x"
        self.split_size = train_val_test_split
        self.workers = num_workers
        self.processer = VideoProcessing(sampling_value, num_frames, image_size)
        self.loader_config = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        self.transform = T.Compose([
            T.Resize(image_size, antialias=True),
            T.ToTensor(),
            T.Normalize(
                mean = [0.485, 0.456, 0.406], 
                std = [0.229, 0.224, 0.225]
            )
        ])



    @property
    def classes(self) -> List[str]:
        return sorted(os.listdir(self.x_path))


    def _processing(self, path: str) -> None:
        """
        process and save processed video
        """
        label = path.replace(self.root, "").split("/")[1]
        folder_path = os.path.join(self.x_path, label)
        file_name = path.split("/")[-1][:-4]
        os.makedirs(folder_path, exist_ok=True)
        video = self.processer(path)
        for i, frame in enumerate(video):
            save_path = os.path.join(folder_path, f"{file_name}_{i}.jpg")
            if not os.path.exists(save_path):
                img = Image.fromarray(frame)
                img.save(save_path)


    def _generate_x_data(self):
        print("[bold]Processing data:[/] Working...", end="\r")
        os.makedirs(self.x_path, exist_ok=True)

        path_list = list(str(video) for video in Path(self.root).rglob(f'*.mpg'))

        with Pool(self.workers) as pool:
            pool.map(self._processing, path_list)
        print("[bold]Processing data:[/] Done      ")


    def prepare_data(self):
        """
        Preprocess data
        """
        if not os.path.exists(self.root):
            raise FileNotFoundError(self.root)
        if not os.path.exists(self.x_path):
            self._generate_x_data()
        else:
            print("[bold]Processing data:[/] Existed")


    def setup(self, stage: str):
        """
        Setup data
        """
        if not hasattr(self, "dataset"):
            self.dataset, self.labels = [], []
            for class_name in self.classes:
                class_dir = os.path.join(self.x_path, class_name)
                image_paths = glob.glob(os.path.join(class_dir, '*.jpg'))
                self.dataset.extend(image_paths)
                self.labels.extend([self.classes.index(class_name)] * len(image_paths))
            X_train, X_val_test, y_train, y_val_test = train_test_split(
                self.dataset, self.labels, 
                train_size = self.split_size[0], 
                stratify = self.labels,
                shuffle = True
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_val_test, y_val_test, 
                test_size = self.split_size[2] / sum(self.split_size[1:]), 
                stratify = y_val_test,
                shuffle = True
            )
            self.train_data = CustomImageDataset(X_train, y_train, self.transform)
            self.val_data = CustomImageDataset(X_val, y_val, self.transform)
            self.test_data = CustomImageDataset(X_test, y_test, self.transform)
        if stage == "fit":
            print(f"[bold]Dataset size:[/] {len(self.dataset):,}")
            print(f"[bold]Number of classes:[/] {len(self.classes):,}")


    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, **self.loader_config, shuffle=True)


    def val_dataloader(self):
        return DataLoader(dataset=self.val_data, **self.loader_config, shuffle=False)


    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, **self.loader_config, shuffle=False)



class UTD_MHADDataModule(LightningDataModule):
    def __init__(
            self, 
            data_path: str, 
            batch_size: int = 32,
            train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
            sampling_value: int = 0,
            num_frames: int = 0,
            image_size: Tuple[int, int] = (224, 224),
            num_workers: int = 0,
            pin_memory: bool = True,
        ):
        super().__init__()
        self.root = data_path
        self.x_path = data_path + "_x"
        self.split_size = train_val_test_split
        self.workers = num_workers
        self.processer = VideoProcessing(sampling_value, num_frames, image_size)
        self.loader_config = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        self.transform = T.Compose([
            T.Resize(image_size, antialias=True),
            T.ToTensor(),
            T.Normalize(
                mean = [0.485, 0.456, 0.406], 
                std = [0.229, 0.224, 0.225]
            )
        ])


    @property
    def classes(self) -> List[str]:
        return os.listdir(self.x_path)


    def _processing(self, path: str) -> None:
        """
        process and save processed video
        """
        labels = { 
            "a1": "Swipe left", 
            "a2": "Swipe right", 
            "a3": "Wave", 
            "a4": "Clap", 
            "a5": "Throw", 
            "a6": "Arm cross", 
            "a7": "Basketball shoot",
            "a8": "Draw X",
            "a9": "Draw circle (forward)",
            "a10": "Draw circle (backward)",
            "a11": "Draw triangle",
            "a12": "Bowling",
            "a13": "Boxing",
            "a14": "Baseball swing",
            "a15": "Tennis swing",
            "a16": "Arm curl",
            "a17": "Tennis serve",
            "a18": "Push",
            "a19": "Knock",
            "a20": "Catch",
            "a21": "Pickup and throw",
            "a22": "Jog",
            "a23": "Walk",
            "a24": "Sit to stand",
            "a25": "Stand to sit",
            "a26": "Lunge",
            "a27": "Squat"
        }
