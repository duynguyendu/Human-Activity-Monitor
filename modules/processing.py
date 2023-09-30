from typing import List, Tuple
import os

import numpy as np
import cv2



class VideoProcessing():
    """
    Process video including:
    - load: 
        Load video and return list of frames

    - sampling: 
        Choose 1 frame every n frames

    - balancing: 
        Decide max or min frame of a video

    - resize: 
        Change size of a video
    
    - auto: 
        Auto apply: Load -> Sampling -> Balancing -> Resize
    """

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
        """
        Object call, apply auto(x) in return
        """
        return self.auto(path)


    def load(self, path: str) -> List[np.ndarray]:
        """
        Load video and return list of frames
        """
        if not os.path.exists(path):
            raise FileExistsError("File not found!")
        video = cv2.VideoCapture(path)
        if not video.isOpened():
            raise RuntimeError("Could not open video file.")
        output = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for _, frame in iter(video.read, (False, None))
        ]
        video.release()
        return output


    def sampling(self, video: np.ndarray, value: int) -> np.ndarray:
        """
        Reduce video size by choose 1 frame every n frame (n = value)
        """
        return video[::value] if value else video


    def balancing(self, video: np.ndarray, value: int) -> np.ndarray:
        """
        Decide number of frame in a video
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
        Change size of a video
        """
        return np.array([cv2.resize(frame, size) for frame in video])


    def auto(self, path: np.ndarray) -> List[np.ndarray]:
        """
        Auto apply: Load -> Sampling -> Balancing -> Resize
        """
        out = self.load(path)
        out = self.sampling(out, self.sampling_value)
        out = self.balancing(out, self.num_frames)
        out = self.resize(out, self.size)
        return out
