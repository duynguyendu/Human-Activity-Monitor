from functools import cached_property
from typing import Tuple, Union
from pathlib import Path
import os

import numpy as np
import cv2

from src.modules.utils import tuple_handler


__all__ = ["Video"]


class Video:
    def __init__(self, path: str, resolution: Tuple = None) -> None:
        """
        Initializes a Video object.

        Args:
            path (str): Path of the video to open.

        Raises:
            FileExistsError: If the file is not found.
        """
        if not os.path.exists(path):
            raise FileExistsError(
                "File not found. Check again or use an absolute path."
            )
        self.video_capture = cv2.VideoCapture(path)
        self.path = path
        self.resolution = tuple_handler(resolution, max_dim=2) if resolution else None
        self.pause = False

    def __iter__(self) -> "Video":
        """
        Initialize video iteration.

        Returns:
            Video: The video object.
        """
        self.current_frame = None
        return self

    def __next__(self) -> Union[cv2.Mat, np.ndarray]:
        """
        Get the next frame from the video.

        Returns:
            MatLike: The next frame.

        Raises:
            StopIteration: When there are no more frames.
        """
        ret, frame = self.video_capture.read()
        if not ret:
            raise StopIteration
        if self.resolution:
            frame = cv2.resize(frame, self.resolution)
        self.current_frame = frame
        return self.current_frame

    def __len__(self) -> int:
        """
        Get the total number of frames in the video.

        Returns:
            int: Total number of frames.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

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
        return self.video_capture

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

    def add_box(
        self,
        top_left: Tuple,
        bottom_right: Tuple,
        color: Tuple = (255, 255, 255),
        thickness: int = 1,
    ) -> None:
        """
        Add a rectangle to the current frame.

        Args:
            top_left (Tuple): Top-left corner coordinates (x, y).
            bottom_right (Tuple): Bottom-right corner coordinates (x, y).
            color (Tuple, optional): Color of the rectangle (B, G, R). Defaults to (255, 255, 255).
            thickness (int, optional): Thickness of the rectangle outline. Defaults to 1.

        Returns:
            None
        """
        cv2.rectangle(
            img=self.current_frame,
            pt1=tuple_handler(top_left, max_dim=2),
            pt2=tuple_handler(bottom_right, max_dim=2),
            color=tuple_handler(color, max_dim=3),
            thickness=int(thickness),
        )

    def add_circle(
        self,
        center: Tuple,
        radius: int,
        color: Tuple = (255, 255, 255),
        thickness: int = 1,
    ) -> None:
        """
        Add a circle to the current frame.

        Args:
            center (Tuple): Center coordinates (x, y).
            radius (int): Circle radius.
            color (Tuple, optional): Color of the circle (B, G, R). Defaults to (255, 255, 255).
            thickness (int, optional): Thickness of the circle outline. Defaults to 1.

        Returns:
            None
        """
        cv2.circle(
            img=self.current_frame,
            center=tuple_handler(center, max_dim=2),
            radius=int(radius),
            color=tuple_handler(color, max_dim=3),
            thickness=int(thickness),
        )

    def add_point(
        self, center: Tuple, radius: int, color: Tuple = (255, 255, 255)
    ) -> None:
        """
        Add a point to the current frame.

        Args:
            center (Tuple): Center coordinates (x, y).
            radius (int): Circle radius.
            color (Tuple, optional): Color of the point (B, G, R). Defaults to (255, 255, 255).

        Returns:
            None
        """
        cv2.circle(
            img=self.current_frame,
            center=tuple_handler(center, max_dim=2),
            radius=int(radius),
            color=tuple_handler(color, max_dim=3),
            thickness=-1,
        )

    def add_text(
        self,
        text: str,
        pos: Tuple,
        font_scale: int = 1,
        color: Tuple = (255, 255, 255),
        thickness: int = 1,
    ) -> None:
        """
        Add text to the current frame.

        Args:
            text (str): Text to add.
            pos (Tuple): Position coordinates (x, y).
            font_scale (int, optional): Font scale for the text. Defaults to 1.
            color (Tuple, optional): Color of the text (B, G, R). Defaults to (255, 255, 255).
            thickness (int, optional): Thickness of the text. Defaults to 1.

        Returns:
            None
        """
        cv2.putText(
            img=self.current_frame,
            text=str(text),
            org=tuple_handler(pos, max_dim=2),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=int(font_scale),
            color=tuple_handler(color, max_dim=3),
            thickness=int(thickness),
        )

    def show(self) -> None:
        """Show the frame"""
        cv2.imshow(self.stem, self.current_frame)

    def release(self) -> None:
        """Release capture"""
        self.video_capture.release()
