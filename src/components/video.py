from functools import cached_property
from typing import Dict, Tuple, Union
from collections import deque
from datetime import datetime
import itertools
import signal
import math
import time
import os
import sys
import fcntl
import traceback

from rich import print
from tqdm import tqdm
import numpy as np
import cv2

from .utils import tuple_handler
from . import Backbone


class Video:
    def __init__(
        self,
        path: str,
        speed: int = 1,
        delay: int = 1,
        subsampling: int = 1,
        sync: bool = True,
        show: bool = True,
        resolution: Tuple = None,
        progress_bar: bool = True,
        show_fps: Union[Dict, bool] = None,
        record: Union[Dict, bool] = None,
    ) -> None:
        """
        Initializes a Video object.

        Args:
            path (str): Path of the video to open.
            speed (int, optional): Playback speed of the video. Defaults to 1.
            delay (int, optional): Delay between frames in milliseconds. Defaults to 1.
            subsampling (int, optional): Skip frames during processing. Defaults to 1.
            sync (bool, optional): Synchronize video playback and frame processing. Defaults to True.
            show (bool, optional): Show video playback. Defaults to True.
            resolution (Tuple, optional): Change resolution of the video. Defaults to None.
            progress_bar (bool, optional): Display progress bar during video playback. Defaults to True.
            show_fps (Dict or bool, optional): Display video real-time FPS. Default to None.
            record (Dict or bool, optional): Record the video. Default to None.

        Raises:
            FileExistsError: If the file is not found.
        """
        # if not os.path.exists(path):
        #     raise FileExistsError(
        #         "File not found. Check again or use an absolute path."
        #     )
        self.path = str(path)
        self.video_capture = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.is_camera = bool(self.total_frame == -1)
        self.__check_speed(speed)
        self.wait = int(delay)
        self.subsampling = max(1, int(subsampling))
        self.sync = bool(sync)
        self.is_show = bool(show)
        self.resolution = tuple_handler(resolution, max_dim=2) if resolution else None
        self.__setup_progress_bar(show=progress_bar)
        self.__setup_fps_display(config=show_fps)
        self.__setup_recorder(config=record)
        self.stop = False
        self.current_progress = 0

    def __check_speed(self, value: Union[int, float]) -> None:
        """
        Check and setup speed parameter.

        Args:
            value (Union[int, float]): Speed value to check.
        """
        if self.is_camera:
            self.speed = 1
        else:
            self.speed = int(max(1, value))
            if isinstance(value, float):
                self.speed_mul = value / self.speed

    def __setup_progress_bar(self, show: bool) -> None:
        """
        Initializes and sets up a progress bar using tqdm.

        Args:
            show (bool): Flag to determine whether to show the progress bar.
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

    def __setup_fps_display(self, config: Union[Dict, bool]) -> None:
        """
        Setup for display video FPS

        Args:
            config (Dict or bool): Configuration or `True` for default
        """
        if config not in [False, None]:
            self.fps_history = deque(maxlen=config.get("smoothness", 30))
            self.fps_pos = tuple_handler(config.get("position", (20, 40)), max_dim=2)

    def __setup_recorder(self, config: Union[Dict, bool]) -> None:
        """
        Setup for record the video.

        Args:
            config (Dict or bool): A dictionary of configurations. False for disable.
        """

        # Disable if config is not provided
        if not config:
            return

        # Set save folder
        save_folder = os.path.join(
            config["path"],
            datetime.now().strftime("%d-%m-%Y") if self.is_camera else self.stem,
        )

        # Create save folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)

        # Config writer
        save_path = os.path.join(save_folder, config["name"] + ".mp4")

        codec = cv2.VideoWriter_fourcc(*"mp4v")

        fps = float(config["fps"] if config["fps"] else self.fps)

        self.recorder_res = (
            tuple_handler(config["resolution"], max_dim=2)
            if config["resolution"]
            else self.size()
        )

        # Config writer
        self.recorder = cv2.VideoWriter(
            filename=save_path, fourcc=codec, fps=fps, frameSize=self.recorder_res
        )

        # Logging
        print(f"[INFO] [bold]Save recorded video to:[/] [green]{save_path}[/]")

    def __resync(func):
        """Synchronize video speed with fps"""

        # Create wrapper function
        def wrapper(self):
            # Check on first run
            if not hasattr(self, "start_time"):
                self.start_time = time.time()

            # Run the function
            output = func(self)

            # Get delay time
            delay = time.time() - self.start_time

            # Check if sync is enable
            if self.sync:
                # Calculate sync value
                sync_time = (
                    1 / self.fps / (self.speed_mul if hasattr(self, "speed_mul") else 1)
                )
                # Apply sync if needed
                if delay < sync_time:
                    time.sleep(sync_time - delay)

            # Initialize debug line
            video_debug = ""

            # Display fps if specified
            if hasattr(self, "fps_history"):
                self.fps_history.append(math.ceil(1 / (time.time() - self.start_time)))
                video_debug += f"FPS: {math.ceil(np.mean(self.fps_history))}"

            # Display latency if specified
            if hasattr(self, "backbone") and self.backbone.show_latency:
                video_debug += f" Latency: {self.backbone.get_latency():.2}s"

            # Display video debug info if specified
            if video_debug:
                self.add_text(
                    text=video_debug, pos=self.fps_pos, thickness=2, color=(0, 0, 255)
                )

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
            try:
                for _, frame in iter(self.video_capture.read, (False, None)):
                    print("frame1")
                    yield frame
            except Exception as ex:
                traceback.print_exception(type(ex), ex, ex.__traceback__)

        # Generate frame queue
        self.queue = itertools.islice(generate(), 0, None, self.speed)

        # Initialize
        self.current_progress = 0
        self.pause = False

        return self

    @__resync
    def __next__(self) -> Union[cv2.Mat, np.ndarray]:
        """
        Get the next frame from the video.

        Returns:
            MatLike: The next frame.
        """

        # Get current frame
        ret, new_frame = self.video_capture.read()
        while not ret:
            print("oh no, missing frame, fml, creating new connection then")
            self.video_capture.release()
            self.video_capture = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
            ret, new_frame = self.video_capture.read()
        
        self.current_frame = new_frame

        # Change video resolution
        if self.resolution:
            self.current_frame = cv2.resize(self.current_frame, self.resolution)

        # Backbone process
        if hasattr(self, "backbone"):
            # Check subsampling
            if (self.current_progress % self.subsampling == 0) and (
                self.backbone.is_free()
            ):
                # Process the current frame
                self.backbone.process(frame=self.current_frame)

            # Apply to current frame
            self.current_frame = self.backbone.apply(self.current_frame)

        # Recorder the video
        if hasattr(self, "recorder"):
            self.recorder.write(
                cv2.resize(self.current_frame, self.recorder_res)
                if self.recorder_res
                else self.current_frame
            )

        # Update progress
        self.current_progress += 1
        self.progress.update(
            max(1, min(self.speed, self.total_frame - self.current_progress))
        )

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
        return self.path.split("/")[-1]

    @cached_property
    def stem(self) -> str:
        """
        Return name of the video without extension

        Returns:
            str: name of the video without extension
        """
        return self.name.split(".")[0]

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
            "tracker": "3",
            "heatmap": "4",
            "track_box": "5",
        }

    def setup_backbone(self, config: Dict) -> None:
        """
        Initializes and sets up backbone for video process.

        Args:
            config (Dict): Configuration for the backbone
        """
        if config["backbone"]:
            self.backbone = Backbone(
                video=self, process_config=config, **config["backbone"]
            )

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
            self.resolution
            if self.resolution
            else (
                int(self.cap.get(prop))
                for prop in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT]
            )
        )
        return (w, h) if not reverse else (h, w)

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
        cv2.namedWindow(self.stem, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(self.stem, self.current_frame)

    def signal_handler(self, sig, frame):
        print("[INFO] [bold]Keyboard Interrupted:[/] [red]Exiting...[/]")
        self.stop = True

    def run(self) -> None:
        """Runs the video playback loop"""
        signal.signal(signal.SIGINT, self.signal_handler)

        while not self.stop:
            self.__next__()
            
            # handle capture command: `capture <filename>`
            data = self.read_stdin_nonblocking()
            if data:
                print("receive data", data)
                commands = data.splitlines()
                for c in commands:
                    args = c.split()
                    if len(args) != 0 and args[0] == "capture":
                        print("capturing image to", args[1])
                        try:
                            cv2.imwrite(args[1], self.current_frame)
                        except Exception as ex:
                            traceback.print_exception(type(ex), ex, ex.__traceback__)

        self.release()

    def release(self) -> None:
        """Release capture"""
        self.video_capture.release()
        if hasattr(self, "recorder"):
            self.recorder.release()
        if hasattr(self, "backbone"):
            self.backbone.finish()
        cv2.destroyAllWindows()
        
    def read_stdin_nonblocking(self):
        fd = sys.stdin.fileno()  # Get the file descriptor for stdin
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)  # Get current file descriptor flags
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)  # Set non-blocking mode

        try:
            data = sys.stdin.read()  # Attempt to read data
        except Exception:  # Handle case where no data is available
            data = ""
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags)  # Restore original flags (optional)

        return data
