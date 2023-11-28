from functools import cached_property
from typing import Dict, Tuple, Union
from pathlib import Path
from queue import Queue
import threading
import itertools
import time
import os

from rich import print
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
        subsampling: int = 1,
        sync: bool = False,
        resolution: Tuple = None,
        progress_bar: bool = True,
    ) -> None:
        """
        Initializes a Video object.

        Args:
            path (str): Path of the video to open.
            speed (int, optional): Playback speed of the video. Defaults to 1.
            delay (int, optional): Delay between frames in milliseconds. Defaults to 1.
            subsampling (int, optional): Skip frames during processing. Defaults to 1.
            sync (bool, optional): Synchronize video playback and frame processing. Defaults to False.
            resolution (Tuple, optional): Change resolution of the video. Defaults to None.
            progress_bar (bool, optional): Display progress bar during video playback. Defaults to True.

        Raises:
            FileExistsError: If the file is not found.
        """
        if not os.path.exists(path):
            raise FileExistsError(
                "File not found. Check again or use an absolute path."
            )
        self.path = path
        self.video_capture = cv2.VideoCapture(path)
        self.__check_speed(speed)
        self.wait = int(delay)
        self.subsampling = max(1, subsampling)
        self.sync = bool(sync)
        self.resolution = tuple_handler(resolution, max_dim=2) if resolution else None
        self.progress_bar = bool(progress_bar)

    def __check_speed(self, value: Union[int, float]) -> None:
        """
        Check and setup speed parameter.

        Args:
            value (Union[int, float]): Speed value to check.
        """
        self.speed = int(max(1, value))
        if isinstance(value, float):
            self.speed_mul = value / self.speed

    def __resync(func):
        """Synchronize video speed with fps"""

        # Create wrapper function
        def wrapper(self):
            # Check if sync is disable
            if not self.sync:
                return func(self)

            # Check on first run
            if not hasattr(self, "start_time"):
                self.start_time = time.time()

            # Run the function
            output = func(self)

            # Get delay time
            delay = time.time() - self.start_time
            # Calculate sync value
            sync_time = (
                1 / self.fps / (self.speed_mul if hasattr(self, "speed_mul") else 1)
            )
            # Apply sync if needed
            if delay < sync_time:
                time.sleep(sync_time - delay)
            # Setup for new circle
            self.start_time = time.time()

            # Return function output
            return output

        return wrapper

    def __iter__(self) -> "Video":
        """
        Initialize video iteration.

        Returns:
            Video: The video object.
        """

        # Video iteration generate
        def generate():
            for _, frame in iter(self.video_capture.read, (False, None)):
                yield frame

        # Generate frame queue
        self.queue = itertools.islice(generate(), 0, None, self.speed)

        # Initialize
        self.pause = False

        # Safe thread to store backbone output
        self.backbone_result = Queue()

        # Setup progress bar
        self.setup_progress_bar(show=self.progress_bar)

        print(f"[bold]Video progress:[/]")

        return self

    @__resync
    def __next__(self) -> Union[cv2.Mat, np.ndarray]:
        """
        Get the next frame from the video.

        Returns:
            MatLike: The next frame.
        """

        # Get current frame
        self.current_frame = next(self.queue)

        # Change video resolution
        if self.resolution:
            self.current_frame = cv2.resize(self.current_frame, self.resolution)

        # Backbone process
        if hasattr(self, "backbone"):
            # Check subsampling
            if (self.progress.n % self.subsampling) == 0:
                # Create a new thread if the backbone process does not exist or is not running
                if (
                    not hasattr(self, "backbone_process")
                    or not self.backbone_process.is_alive()
                ):
                    # Spam a new thread
                    self.backbone_process = threading.Thread(
                        target=self.backbone.apply,
                        args=(self.current_frame, self.backbone_result),
                        daemon=True,
                    )

                    # Start running the thread
                    self.backbone_process.start()

                # Retrieve a new overlay and mask if any threads have completed
                if not self.backbone_result.empty():
                    self.overlay = self.backbone_result.get()
                    self.mask = cv2.cvtColor(self.overlay, cv2.COLOR_BGR2GRAY) != 0

            # Apply mask to current frame
            if hasattr(self, "overlay"):
                self.current_frame[self.mask] = self.overlay[self.mask]

        # Recorder the video
        if hasattr(self, "recorder"):
            self.recorder.write(
                cv2.resize(self.current_frame, self.recorder_res)
                if self.recorder_res
                else self.current_frame
            )

        # Update progress
        self.progress.update(min(self.speed, self.total_frame - self.progress.n))

        # Sync frame speed
        if self.sync:
            self.__resync()

        # Return current frame
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

    @cached_property
    def shortcuts(self) -> Dict:
        """
        Return shortcut of the video

        Returns:
            Dict: Shortcut of the video
        """
        return {
            "quit": "q",
            "pause": "p",
            "resume": "r",
            "detector": "1",
            "classifier": "2",
            "heatmap": "3",
            "track_box": "4",
        }

    def setup_backbone(self, config: Dict) -> None:
        """
        Initializes and sets up backbone for video process.

        Args:
            config (Dict): Configuration for the backbone
        """
        self.backbone = Backbone(video=self, config=config)

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
            smoothing=0.3,
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

    def custom_shortcut(self, values: Dict):
        """
        Updates the existing shortcuts dictionary with the provided new_shortcuts.

        Args:
            new_shortcuts (Dict): Dictionary containing shortcut name-key pairs.
        """
        self.shortcuts.update(
            {name: key for name, key in values.items() if name in self.shortcuts}
        )

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

        self.recorder_res = tuple_handler(resolution, max_dim=2) if resolution else None

        # Config writer
        self.recorder = cv2.VideoWriter(
            filename=self.recorder_path,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=self.recorder_fps,
            frameSize=self.recorder_res,
        )

        print(f"  [bold]Save recorded video to:[/] [green]{self.recorder_path}.mp4[/]")

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
        if not hasattr(self, "current_frame"):
            raise ValueError(
                "No current frame to show. Please run or loop through the video first."
            )
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
            True
            if key == ord(self.shortcuts["pause"])
            else False
            if key == ord(self.shortcuts["resume"])
            else self.pause
        )

        # Check features toggle
        if hasattr(self, "backbone"):
            for process in self.backbone.status:
                if process != "human_count" and key == ord(self.shortcuts[process]):
                    self.backbone.status[process] = not self.backbone.status[process]

        # Check continue
        return True if not key == ord("q") else False

    def release(self) -> None:
        """Release capture"""
        self.video_capture.release()
        if hasattr(self, "recorder"):
            self.recorder.release()
        if hasattr(self, "backbone"):
            self.backbone.finish()
        cv2.destroyWindow(self.stem)
