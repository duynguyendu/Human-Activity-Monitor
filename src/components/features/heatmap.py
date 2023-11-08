from typing import Tuple
from pathlib import Path

import numpy as np
import cv2

from src.modules.utils import tuple_handler


class Heatmap:
    def __init__(self, shape: Tuple[int, int]) -> None:
        """
        Initializes a Heatmap object.

        Args:
            layer (np.ndarray): starting layer
        """
        self.layer = np.zeros(shape=tuple_handler(shape, max_dim=2), dtype=np.uint8)

    def update(self, area: Tuple, value: int) -> np.ndarray:
        """
        Update map layer

        Args:
            area (int): area to increase
            value (_type_): amount to update
            blurriness (float, optional): the blurriness of the heat layer. Defaults to 1.0

        Returns:
            np.ndarray: layer after updated
        """

        # Grow
        x1, y1, x2, y2 = tuple_handler(area, max_dim=4)
        x1, y1 = int(x1 * 0.95), int(y1 * 0.95)
        x2, y2 = int(x2 * 1.05), int(y2 * 1.05)
        self.layer[y1:y2, x1:x2] = np.minimum(
            self.layer[y1:y2, x1:x2] + value, 255 - value
        )

    def decay(self, value: int = 1) -> None:
        """
        Reduce heatmap value

        Args:
            value (int): percentage to decrease
        """
        self.layer = ((1 - value / 100) * self.layer).astype(np.uint8)

    def apply(
        self,
        image: np.ndarray,
        blurriness: float = 1.0,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Apply heatmap

        Args:
            image (np.ndarray): image to apply heatmap
            layer (np.ndarray): layer to apply heat
            alpha (float, optional): Opacity of the heat layer. Defaults to 0.5

        Returns:
            np.ndarray: result image
        """

        # Blur
        blurriness = int(blurriness * 100)
        blurriness = blurriness + 1 if blurriness % 2 == 0 else blurriness
        self.layer = cv2.stackBlur(self.layer, (blurriness, blurriness), 0)

        # Apply heat to layer
        heatmap = cv2.applyColorMap(self.layer, cv2.COLORMAP_TURBO)

        # Combined image and heat layer
        cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0, image)

        # Check if save video
        if hasattr(self, "_video_writer"):
            self._video_writer.write(heatmap)

        # Check if save image
        if hasattr(self, "_image_writer"):
            self._image_writer["count"] += 1
            self._image_writer["image"] += heatmap
            cv2.imwrite(
                self._image_writer["path"],
                (self._image_writer["image"] / self._image_writer["count"]).astype(
                    np.uint8
                ),
            )

        return image, heatmap

    def save_video(
        self, save_path: str, fps: int, size: Tuple, codec: str = "mp4v"
    ) -> None:
        """
        Create a video writer for heatmap

        Args:
            save_path (str): path to store writed video.
            fps (int): FPS of output video.
            size (Tuple): size of output video.
            codec (str, optional): codec for write video. Defaults to "mp4v".

        Returns:
            None
        """
        save_path = Path(save_path)

        # Create save folder
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Create video writer
        self._video_writer = cv2.VideoWriter(
            filename=str(save_path),
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=fps,
            frameSize=size,
        )

    def save_image(self, save_path: str, size: Tuple) -> None:
        """_summary_

        Args:
            save_path (str): _description_
            size (Tuple): _description_
        """
        save_path = Path(save_path)

        #  Create save folder
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self._image_writer = {
            "path": str(save_path),
            "count": 0,
            "image": np.zeros(tuple_handler((*size, 3), max_dim=3), dtype=np.float32),
        }

    def release(self):
        """Release capture"""
        if hasattr(self, "writer"):
            self.writer.release()
