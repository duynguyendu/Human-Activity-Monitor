from typing import Tuple, Generator
from pathlib import Path
import os

from cv2.typing import MatLike
import cv2


__all__ = ["Video"]


class Video:
    """
    Video handle class

    Methods:
        open(path): open video
        get_frame(cap): return a frames Generator
        get_size(cap): return video size
        get_total_frame(cap): return total frame
        get_fps(cap): return fps
        writer(save_path, fps, size, codec): return video writer
        end(*args): close up
    """

    @staticmethod
    def open(path: str) -> cv2.VideoCapture:
        """
        Open video

        Args:
            path (str): path of video to open

        Raises:
            FileExistsError: if file not found

        Returns:
            cv2.VideoCapture: video capture
        """
        if not os.path.exists(path):
            raise FileExistsError("File not found. Check again or use absolute path.")
        return cv2.VideoCapture(path)

    @staticmethod
    def get_frame(cap: cv2.VideoCapture) -> Generator[MatLike, None, None]:
        """
        Return a frames Generator

        Args:
            cap (cv2.VideoCapture): video capture

        Yields:
            MatLike: each video frame
        """
        for _, frame in iter(cap.read, (False, None)):
            yield frame

    @staticmethod
    def get_size(cap: cv2.VideoCapture) -> Tuple[int, int]:
        """
        Return video size

        Args:
            cap (cv2.VideoCapture): video capture

        Returns:
            Tuple: size of the video (Width, Height)
        """
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @staticmethod
    def get_total_frame(cap: cv2.VideoCapture) -> int:
        """
        Return total number of frame

        Args:
            cap (cv2.VideoCapture): video capture

        Returns:
            int: total frame
        """
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @staticmethod
    def get_fps(cap: cv2.VideoCapture) -> int:
        """
        Return FPS of the video

        Args:
            cap (cv2.VideoCapture): video capture

        Returns:
            int: FPS of the video
        """
        return int(cap.get(cv2.CAP_PROP_FPS))

    def writer(
        save_path: str, fps: int, size: Tuple, codec: str = "mp4v"
    ) -> cv2.VideoWriter:
        """
        Create a video writer

        Args:
            save_path (str): path to store writed video.
            fps (int): FPS of output video.
            size (Tuple): size of output video.
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
            fps=fps,
            frameSize=size,
        )

    def end(*args: cv2.VideoCapture) -> None:
        """Close up all video process"""
        [cap.release() for cap in args]
        cv2.destroyAllWindows()
