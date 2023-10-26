from typing import Tuple
import os

import numpy as np
import cv2



class VideoProcessing:
    """
    Process included:

    1. load: Load video and return list of frames.
    2. Sampling: Choose 1 frame every n frames.
    3. Truncating: Equally trim both head and tail if video length > max_frame.
    4. Padding: Pad black frame to end of video if video length < min_frame.
    5. Add border: Adds a border around a given video frame to make it a square frame.
    6. Resize: Change size of a video.

    - Auto: Auto apply: Load -> Sampling -> Truncating -> Padding -> Add border -> Resize
    """
    def __init__(
            self,
            sampling_value: int = 0,
            max_frame: int = 0,
            min_frame: int = 0,
            add_border: bool = False,
            size: Tuple[int, int] = 0
        ) -> None:
        """
        Initialize the video processing operations.

        Args:
            sampling_value (int, optional): The sampling value for frame selection. Default: 0
            max_frame (int, optional): The maximum number of frames to retain. Default: 0
            min_frame (int, optional): The minimum number of frames required. Default: 0
            add_border (bool, optional): Whether to add borders to frames. Default: False
            size (Tuple[int, int], optional): The target size for resizing frames. Default: 0
        """
        self.sampling_value = sampling_value
        self.max_frame = max_frame
        self.min_frame = min_frame
        self.border = add_border
        self.size = size


    def __call__(self, path: str) -> np.ndarray:
        """
        Auto apply: Load -> Sampling -> Truncating -> Padding -> Add border -> Resize

        Args:
            path (np.ndarray): Path of video to be processed.

        Returns:
            np.ndarray: A NumPy array representing the processed video.
        """
        return self.auto(path)


    def load(self, path: str) -> np.ndarray:
        """
        Load a video file and return it as a NumPy array.

        Args:
            path (str): The file path to the video.

        Returns:
            np.ndarray: A NumPy array representing the video frames.

        Raises:
            - FileExistsError: If the specified file does not exist.
            - RuntimeError: If the video file cannot be opened.
        """
        if not os.path.exists(path):
            raise FileExistsError("File not found!")
        video = cv2.VideoCapture(path)
        if not video.isOpened():
            raise RuntimeError("Could not open video file.")
        output = np.array([
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for _, frame in iter(video.read, (False, None))
        ])
        video.release()
        return output


    def sampling(self, video: np.ndarray, value: int) -> np.ndarray:
        """
        Perform frame sampling on a video.

        Args:
            video (np.ndarray): Input video as a NumPy array.
            value (int): The sampling value. If 0, no sampling is performed.

        Returns:
            np.ndarray: A NumPy array representing the sampled video frames.
        """
        return video[::value] if value else video


    def truncating(self, video: np.ndarray, max_frame: int) -> np.ndarray:
        """
        Truncate a video to a specified maximum number of frames.

        Args:
            video (np.ndarray): Input video as a NumPy array.
            max_frame (int): The maximum number of frames to retain.

        Returns:
            np.ndarray: A NumPy array representing the truncated video frames.
        """
        middle_frame = len(video) // 2
        m = max_frame // 2
        r = max_frame % 2
        return video[middle_frame - m : middle_frame + m + r]


    def padding(self, video: np.ndarray, min_frame: int) -> np.ndarray:
        """
        Pad a video with black frames to meet a minimum frame length.

        Args:
            video (np.ndarray): Input video as a NumPy array.
            min_frame (int): The desired minimum length of the video in frames.

        Returns:
            np.ndarray: A NumPy array representing the video after padding.
        """
        zeros_array = np.zeros((min_frame, *video.shape[1:]), dtype=np.uint8)
        zeros_array[:len(video), ...] = video
        return zeros_array


    def add_border(self, video: np.ndarray, border_color: Tuple|int = (0, 0, 0)) -> np.ndarray:
        """
        Adds a border around a given video frame to make it a square frame.

        Args:
            video (np.ndarray): Input video frame as a NumPy array.
            border_color (Tuple|int): Color of the border. Defaults to (0, 0, 0) for black.

        Returns:
            np.ndarray: A NumPy array representing the video after add border.
        """
        img_h, img_w = video.shape[:2]
        target_size = max(img_h, img_w)

        border_v = (target_size - img_h) // 2
        border_h = (target_size - img_w) // 2

        border = lambda x: cv2.copyMakeBorder(x, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, border_color)
        return np.array([border(frame) for frame in video])


    def resize(self, video: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Resize each frame in a video to the specified dimensions.

        Args:
            video (np.ndarray): Input video as a NumPy array.
            size (Tuple[int, int]): A tuple specifying the target size (width, height) for each frame.

        Returns:
            np.ndarray: A NumPy array representing the resized video.
        """
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, (tuple, list)) and len(size) == 2:
            size = tuple(size)
        else:
            raise TypeError(f"The 'size' parameter must be an int and > 0 or tuple or list with length == 2")
        return np.array([cv2.resize(frame, size) for frame in video])


    def auto(self, path: str) -> np.ndarray:
        """
        Auto apply: Load -> Sampling -> Truncating -> Padding -> Add border -> Resize

        Args:
            path (np.ndarray): Path of video to be processed.

        Returns:
            np.ndarray: A NumPy array representing the processed video.
        """
        # Load
        video = self.load(path)
        # Sampling
        if self.sampling_value > 0:
            video = self.sampling(video, self.sampling_value)
        # Truncating
        if self.max_frame > 0:
            video = self.truncating(video, self.max_frame)
        # Padding
        if self.min_frame > 0:
            video = self.padding(video, self.min_frame)
        # Add border
        if self.border:
            video = self.add_border(video)
        # Resize
        if self.size:
            video = self.resize(video, self.size)
        return video
