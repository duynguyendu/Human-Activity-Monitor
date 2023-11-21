from functools import cached_property
from typing import Tuple, Union
from pathlib import Path
import itertools
import os

from tqdm import tqdm
import numpy as np
import cv2

from src.modules.utils import tuple_handler
from . import Backbone


class Video:
    def __init__(
        self,
        path: str,
        speed: int = 1,
        delay: int = 1,
        resolution: Tuple = None,
        backbone: Backbone = None,
        progress_bar: bool = True,
    ) -> None:
        """
        Initializes a Video object.

        Args:
            path (str): Path of the video to open.
            speed (int, optional): Playback speed of the video. Defaults to 1.
            delay (int, optional): Delay between frames in milliseconds. Defaults to 1.
            resolution (Tuple, optional): Change resolution of the video.
            backbone (Backbone, optional): Backbone for video processing. Defaults to None.
            progress_bar (bool, optional): Display progress bar during video playback. Defaults to True.

        Raises:
            FileExistsError: If the file is not found.
        """
        if not os.path.exists(path):
            raise FileExistsError(
                "File not found. Check again or use an absolute path."
            )
        self.video_capture = cv2.VideoCapture(path)
        self.path = path
        self.speed = speed
        self.wait = delay
        self.resolution = tuple_handler(resolution, max_dim=2) if resolution else None
        self.backbone = backbone
        self.progress_bar = progress_bar
        self.pause = False

    def __iter__(self) -> "Video":
        """
        Initialize video iteration.

        Returns:
            Video: The video object.
        """
        self.current_frame = None
        self.setup_progress_bar(show=self.progress_bar)

        def generate():
            for _, frame in iter(self.video_capture.read, (False, None)):
                yield frame

        self.queue = itertools.islice(generate(), 0, None, self.speed)
        return self

    def __next__(self) -> Union[cv2.Mat, np.ndarray]:
        """
        Get the next frame from the video.

        Returns:
            MatLike: The next frame.
        """
        frame = next(self.queue)

        # Change video resolution
        if self.resolution:
            frame = cv2.resize(frame, self.resolution)

        # Backbone process
        if self.backbone:
            frame = self.backbone.process(frame)

        # Recorder the video
        if hasattr(self, "recorder"):
            self.recorder.write(cv2.resize(frame, self.recorder_res))

        # Update progress
        if (self.progress.n + self.speed) > self.total_frame:
            # Handle progress overflow
            self.progress.update(self.total_frame - self.progress.n)
        else:
            self.progress.update(self.speed)

        # Return current frame
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

    def setup_progress_bar(self, show: bool) -> None:
        """
        Initializes and sets up a progress bar using tqdm.

        Args:
            show (bool): Flag to determine whether to show the progress bar.

        Returns:
            None
        """
        self.progress = tqdm(
            disable=not show,
            total=self.total_frame,
            desc=f"  {self.name}",
            unit=" frame",
            smoothing=0.01,
            delay=0.1,
            colour="cyan",
        )

    def custom_progress_bar(self, tqdm: tqdm) -> None:
        """
        Sets a custom tqdm progress bar for the instance.

        Args:
            tqdm: An instance of the tqdm class representing a custom progress bar.
        """
        self.progress = tqdm

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

    def record(
        self,
        save_path: str = "records",
        save_name: str = "output",
        fps: int = None,
        resolution: Tuple = None,
        codec: str = "mp4v",
    ) -> cv2.VideoWriter:
        """
        Record the video.

        Args:
            save_path (str, optional): Path to store the written video. Defaults to "records".
            save_name (str, optional): Name of the output video file. Defaults to "output".
            fps (int, optional): Frames per second for the video. If None, it defaults to the original fps of the source video.
            resolution (Tuple, optional): Resolution of the video (width, height). If None, it defaults to the original size of the source video.
            codec (str, optional): Codec for writing the video. Defaults to "mp4v".

        Returns:
            cv2.VideoWriter: Object to write video.
        """
        save_path = Path(save_path)

        # Create save folder
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.recorder_path = os.path.join(save_path, self.stem, save_name + ".mp4")

        self.recorder_fps = int(fps) if fps else self.fps

        self.recorder_res = (
            tuple_handler(resolution, max_dim=2) if resolution else self.size()
        )

        # Config writer
        self.recorder = cv2.VideoWriter(
            filename=self.recorder_path,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=self.recorder_fps,
            frameSize=self.recorder_res,
        )

        return self.recorder

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

    def run(self) -> None:
        """Runs the video playback loop"""
        for _ in self:
            self.show()

            if not self.delay(self.wait):
                break

        self.release()

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

    def release(self) -> None:
        """Release capture"""
        self.video_capture.release()
        if hasattr(self, "recorder"):
            self.recorder.release()
        cv2.destroyWindow(self.stem)
