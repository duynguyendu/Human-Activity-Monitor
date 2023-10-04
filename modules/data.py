from typing import List, Tuple, Union
from multiprocessing import Pool
from pathlib import Path
import random
import shutil
import os

from modules.transform import DataAugmentation
from modules.processing import VideoProcessing

from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder

from lightning.pytorch import LightningDataModule

from rich import print
from PIL import Image


__all__ = [
    "CustomDataModule",
    "UCF11DataModule",
    "UCF50DataModule",
    "UTD_MHADDataModule"
]



class CustomDataModule(LightningDataModule):
    def __init__(
            self, 
            data_path: str, 
            batch_size: int = 32,
            train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
            keep_temp: bool = False,
            sampling_value: int = 0,
            max_frames: int = 0,
            min_frames: int = 0,
            image_size: Tuple[int, int] = (224, 224),
            num_workers: int = 0,
            pin_memory: bool = True,
        ):
        super().__init__()
        self.root = data_path
        self.x_path = data_path + "_x"
        self.temp_path = data_path + "_tmp"
        self.extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.mpg']
        self.split_size = train_val_test_split
        self.workers = num_workers
        self.keep_temp = keep_temp
        self.processer = VideoProcessing(sampling_value, max_frames, min_frames, image_size)
        self.transform = DataAugmentation(image_size)
        self.loader_config = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }


    @property
    def classes(self) -> List[str]:
        return sorted(os.listdir(self.root))


    def _frame_generate(self, path: str) -> None:
        """
        process and save processed video
        """
        file_name = path.split("/")[-1][:-4]
        dst_path = "/".join(path.replace(self.temp_path, self.x_path).split("/")[:-1])
        os.makedirs(dst_path, exist_ok=True)
        video = self.processer(path)
        for i, frame in enumerate(video):
            save_path = os.path.join(dst_path, f"{file_name}_{i}.jpg")
            if not os.path.exists(save_path):
                img = Image.fromarray(frame)
                img.save(save_path)


    def _split_data(self, class_folder: str):
        # Create new path train, val, test folder
        dst_paths = [os.path.join(self.temp_path, folder, class_folder) for folder in ["train", "val", "test"]]
        [os.makedirs(path, exist_ok=True) for path in dst_paths]

        # Get all video path in source folder
        class_path = os.path.join(self.root, class_folder)
        video_paths = [str(video) for ext in self.extensions for video in Path(self.temp_path).rglob("*" + ext)]

        num_videos = len(video_paths)

        random.shuffle(video_paths)

        # Split into train, val, test chunk
        split_sizes = [round(num_videos * size) for size in self.split_size]
        splited_video_paths = [
            video_paths[:split_sizes[0]], 
            video_paths[split_sizes[0]:split_sizes[0]+split_sizes[1]], 
            video_paths[split_sizes[0]+split_sizes[1]:]
        ]

        # Copy to new destination
        for video_paths, split_path in zip(splited_video_paths, dst_paths):
            for video_path in video_paths:
                dst_path = os.path.join(split_path, video_path.split("/")[-1])
                if not os.path.exists(dst_path):
                    shutil.copyfile(video_path, dst_path)


    def prepare_data(self):
        """
        Preprocess data
        """
        if not os.path.exists(self.root):
            raise FileNotFoundError(self.root)

        if not os.path.exists(self.x_path):
            print("[bold]Processing data:[/] Splitting data... ", end="\r")
            class_folders = os.listdir(self.root)
            with Pool(self.workers) as pool:
                pool.map(self._split_data, class_folders)

            print("[bold]Processing data:[/] Generating data...", end="\r")
            video_paths = [str(video) for ext in self.extensions for video in Path(self.temp_path).rglob("*" + ext)]
            with Pool(self.workers) as pool:
                pool.map(self._frame_generate, video_paths)

            shutil.rmtree(self.temp_path) if not self.keep_temp else None
            print("[bold]Processing data:[/] Done              ")

        else:
            print("[bold]Processing data:[/] Existed")


    def setup(self, stage: str):
        """
        Setup data
        """
        if not hasattr(self, "dataset"):
            self.train_data = ImageFolder(os.path.join(self.x_path, "train"), transform=self.transform)
            self.val_data = ImageFolder(os.path.join(self.x_path, "val"), transform=self.transform)
            self.test_data = ImageFolder(os.path.join(self.x_path, "test"), transform=self.transform)
            
            self.dataset = ConcatDataset([self.train_data, self.val_data, self.test_data])

        if stage == "fit":
            print(f"[bold]Dataset size:[/] {len(self.dataset):,}")
            print(f"[bold]Number of classes:[/] {len(self.classes):,}")


    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, **self.loader_config, shuffle=True)


    def val_dataloader(self):
        return DataLoader(dataset=self.val_data, **self.loader_config, shuffle=False)


    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, **self.loader_config, shuffle=False)



class UCF11DataModule(CustomDataModule):
    def __init__(
            self, 
            data_path: str, 
            batch_size: int = 32,
            train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
            keep_temp: bool = False,
            sampling_value: int = 0,
            max_frames: int = 0,
            min_frames: int = 0,
            image_size: Tuple[int, int] = (224, 224),
            num_workers: int = 0,
            pin_memory: bool = True,
        ):
        self.save_hyperparameters()
        super().__init__(**self.hparams)



class UCF50DataModule(CustomDataModule):
    def __init__(
            self, 
            data_path: str, 
            batch_size: int = 32,
            train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
            keep_temp: bool = False,
            sampling_value: int = 0,
            max_frames: int = 0,
            min_frames: int = 0,
            image_size: Tuple[int, int] = (224, 224),
            num_workers: int = 0,
            pin_memory: bool = True,
        ):
        self.save_hyperparameters()
        super().__init__(**self.hparams)



class HMDB51DataModule(CustomDataModule):
    def __init__(
            self, 
            data_path: str, 
            batch_size: int = 32,
            train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
            keep_temp: bool = False,
            sampling_value: int = 0,
            max_frames: int = 0,
            min_frames: int = 0,
            image_size: Tuple[int, int] = (224, 224),
            num_workers: int = 0,
            pin_memory: bool = True,
        ):
        self.save_hyperparameters()
        super().__init__(**self.hparams)



class UTD_MHADDataModule(CustomDataModule):
    LABELS = {
        f"a{i}": action
        for i, action in enumerate(
            [
                "Swipe left", "Swipe right", "Wave", "Clap", "Throw",
                "Arm cross", "Basketball shoot", "Draw X", "Draw circle (forward)",
                "Draw circle (backward)", "Draw triangle", "Bowling", "Boxing",
                "Baseball swing", "Tennis swing", "Arm curl", "Tennis serve", "Push",
                "Knock", "Catch", "Pickup and throw", "Jog", "Walk", "Sit to stand",
                "Stand to sit", "Lunge", "Squat"
            ],
            start=1
        )
    }

    def __init__(
            self, 
            data_path: str, 
            batch_size: int = 32,
            train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
            keep_temp: bool = False,
            sampling_value: int = 0,
            max_frames: int = 0,
            min_frames: int = 0,
            image_size: Tuple[int, int] = (224, 224),
            num_workers: int = 0,
            pin_memory: bool = True,
        ):
        self.save_hyperparameters()
        super().__init__(**self.hparams)


    @property
    def classes(self) -> List[str]:
        return sorted(self.LABELS.values())


    def _split_data(self, data: Tuple[str, List[str]]):
        label, videos = data

        # Create new path train, val, test folder
        dst_paths = [os.path.join(self.temp_path, folder, label) for folder in ["train", "val", "test"]]
        [os.makedirs(path, exist_ok=True) for path in dst_paths]

        # Get all video path in source folder
        video_paths = [os.path.join(self.root, video) for video in videos]

        num_videos = len(video_paths)

        random.shuffle(video_paths)

        # Split into train, val, test chunk
        split_sizes = [round(num_videos * size) for size in self.split_size]
        splited_video_paths = [
            video_paths[:split_sizes[0]], 
            video_paths[split_sizes[0]:split_sizes[0]+split_sizes[1]], 
            video_paths[split_sizes[0]+split_sizes[1]:]
        ]

        # Copy to new destination
        for video_paths, split_path in zip(splited_video_paths, dst_paths):
            for video_path in video_paths:
                dst_path = os.path.join(split_path, video_path.split("/")[-1])
                if not os.path.exists(dst_path):
                    shutil.copyfile(video_path, dst_path)


    def prepare_data(self):
        """
        Preprocess data
        """
        if not os.path.exists(self.root):
            raise FileNotFoundError(self.root)

        if not os.path.exists(self.x_path):
            print("[bold]Processing data:[/] Splitting data... ", end="\r")
            data = {label: [] for label in self.classes}
            for video in os.listdir(self.root):
                key = video.split("_")[0]
                data[self.LABELS[key]].append(video)

            with Pool(self.workers) as pool:
                pool.map(self._split_data, data.items())

            print("[bold]Processing data:[/] Generating data...", end="\r")
            video_paths = [str(video) for ext in self.extensions for video in Path(self.temp_path).rglob("*" + ext)]
            with Pool(self.workers) as pool:
                pool.map(self._frame_generate, video_paths)

            shutil.rmtree(self.temp_path)
            print("[bold]Processing data:[/] Done              ")

        else:
            print("[bold]Processing data:[/] Existed")
