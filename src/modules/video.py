from functools import cached_property
from typing import Tuple, Generator
from pathlib import Path
import os

from cv2.typing import MatLike
import cv2

from src.modules.utils import tuple_handler


__all__ = ["Video"]


class Video:
    def __init__(self, path: str) -> None:
        """
        Initializes a Video object.

        Args:
            path (str): path of video to open

        Raises:
            FileExistsError: if file not found
        """
        if not os.path.exists(path):
            raise FileExistsError("File not found. Check again or use absolute path.")
        self.path = path
        self.pause = False

    @cached_property
    def name(self) -> str:
        """
        Return name of the video

        Returns:
            str: name of the video
        """
        return str(Path(self.path).name)

    @cached_property
    def stem(self) -> str:
        """
        Return name of the video without extension

        Returns:
            str: name of the video without extension
        """
        return str(Path(self.path).stem)

    @cached_property
    def cap(self) -> cv2.VideoCapture:
        """
        Return video capture

        Returns:
            VideoCapture
        """
        return cv2.VideoCapture(self.path)

    @cached_property
    def total_frame(self) -> int:
        """
        Return total number of frame

        Returns:
            int: total frame
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @cached_property
    def fps(self) -> int:
        """
        Return FPS of the video

        Returns:
            int: FPS of the video
        """
        return int(self.cap.get(cv2.CAP_PROP_FPS))

    def size(self, reverse: bool = False) -> Tuple[int, int]:
        """
        Return video size

        Args:
            reverse (bool): reverse output. Defaults to (Width, Height)

        Returns:
            Tuple: size of the video
        """
        w, h = (
            int(self.cap.get(prop))
            for prop in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT]
        )
        return (w, h) if not reverse else (h, w)

    def get_frame(self) -> Generator[MatLike, None, None]:
        """
        Return a frames Generator

        Yields:
            MatLike: each video frame
        """
        for _, frame in iter(self.cap.read, (False, None)):
            yield frame

    def delay(self, value: int) -> bool:
        """
        Video delay

        Args:
            value (int): millisecond

        Returns:
            bool: True if continue else False
        """
        key = cv2.waitKey(value if not self.pause else 0) & 0xFF

        # Check pause status
        self.pause = (
            True if key == ord("p") else False if key == ord("r") else self.pause
        )

        # Check continue
        return True if not key == ord("q") else False

    def writer(self, save_path: str, codec: str = "mp4v") -> cv2.VideoWriter:
        """
        Create a video writer

        Args:
            save_path (str): path to store writed video.
            codec (str, optional): codec for write video. Defaults to "mp4v".

        Returns:
            cv2.VideoWriter: use to write video
        """
        save_path = Path(save_path)

        # Create save folder
        save_path.parent.mkdir(parents=True, exist_ok=True)

        return cv2.VideoWriter(
            filename=str(save_path),
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=self.fps,
            frameSize=self.size(),
        )

    def release(self) -> None:
        """Release capture"""
        self.cap.release()

    def add_text(
        self,
        frame: MatLike,
        text: str,
        pos: Tuple,
        font_scale: int = 1,
        color: Tuple = 255,
        thickness: int = 2,
    ) -> None:
        """
        Add text to video

        Args:
            frame (MatLike): frame to add text
            text (str): text to add

        Returns:
            None
        """
        cv2.putText(
            img=frame,
            text=str(text),
            org=tuple_handler(pos, max_dim=2),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=tuple_handler(color, max_dim=3),
            thickness=thickness,
        )
