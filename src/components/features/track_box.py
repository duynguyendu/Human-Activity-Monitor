from typing import Dict, Tuple, Union, Iterable
from collections import deque
import cv2

import numpy as np

from src.modules.utils import tuple_handler


class Box:
    def __init__(
        self,
        name: str,
        top_left: Tuple,
        bottom_right: Tuple,
        smoothness: int,
        color: Tuple,
        box_thickness: int,
        text_pos_adjust: Tuple,
        font_scale: int,
        text_thickness: int,
    ) -> None:
        """
        Initialize a box with given top-left and bottom-right coordinates.

         Args:
            name (str): Name of the box.
            top_left (Tuple): Top-left corner coordinates (x, y).
            bottom_right (Tuple): Bottom-right corner coordinates (x, y).
            smoothness (int): Number of frames for smoothing. Defaults to 1.
            color (Tuple): RGB values representing the color of the box and text.
            box_thickness (int): Thickness of the box border.
            text_pos_adjust (Tuple): Adjustment to the top-left corner for text positioning.
            font_scale (int): Font scale for the displayed text.
            text_thickness (int): Thickness of the text.
        """
        self.name = str(name)
        self.xyxy = (*top_left, *bottom_right)
        self.history = deque([], maxlen=smoothness)
        self.count = 0
        self.box_config = {
            "pt1": top_left,
            "pt2": bottom_right,
            "color": color,
            "thickness": box_thickness,
        }
        self.text_config = {
            "org": tuple(x + y for x, y in zip(top_left, text_pos_adjust)),
            "fontScale": font_scale,
            "color": color,
            "thickness": text_thickness,
        }

    def check(self, pos: Tuple) -> None:
        """
        Check if a position is within the box.

        Args:
            pos (Tuple): Position coordinates (x, y).
        """
        x1, y1, x2, y2 = self.xyxy

        X, Y = tuple_handler(pos, max_dim=2)

        if (x1 <= X <= x2) and (y1 <= Y <= y2):
            self.count += 1

    def get_value(self) -> int:
        """
        Get the smoothed count value.

        Returns:
            int: Smoothed count value.
        """
        self.history.append(self.count)
        self.count = 0
        return int(np.mean(self.history))

    def apply(self, image: Union[cv2.Mat, np.ndarray]) -> Union[cv2.Mat, np.ndarray]:
        """
        Apply result to the given image

        Args:
            image (Union[cv2.Mat, np.ndarray]): Input image

        Returns:
            Union[cv2.Mat, np.ndarray]: Result image
        """
        image = cv2.rectangle(image, **self.box_config)
        image = cv2.putText(
            img=image,
            text=str(self.get_value()),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            **self.text_config,
        )
        return image


class TrackBox:
    def __init__(self, default_config: Dict, boxes: Iterable[Dict]) -> None:
        """
        Initialize a TrackBox class with given default configuration

        Args:
            smoothness (int): Number of frames for smoothing. Defaults to 1.
            color (Tuple): RGB values representing the color of the box and text.
            box_thickness (int): Thickness of the box border.
            text_pos_adjust (Tuple): Adjustment to the top-left corner for text positioning.
            font_scale (int): Font scale for the displayed text.
            text_thickness (int): Thickness of the text.
        """
        self.default_config = default_config
        self.boxes = [self.new(box_config) for box_config in boxes]

    def new(self, config: Dict) -> None:
        """
        Create a new tracked box.

        Args:
            config (Dict): Configuration of the box
        """
        return Box(
            **config,
            text_pos_adjust=self.default_config["text"]["position"],
            font_scale=self.default_config["text"]["font_scale"],
            text_thickness=self.default_config["text"]["thickness"],
            box_thickness=self.default_config["box"]["thickness"],
        )

    def check(self, pos: Tuple) -> None:
        """
        Check if the provided position is within any of the tracked boxes.

        Args:
            pos (Tuple): Position coordinates (x, y).
        """
        [box.check(pos) for box in self.boxes]

    def apply(self, image: Union[cv2.Mat, np.ndarray]) -> Union[cv2.Mat, np.ndarray]:
        """
        Apply result to the given image

        Args:
            image (Union[cv2.Mat, np.ndarray]): Input image

        Returns:
            Union[cv2.Mat, np.ndarray]: Result image
        """
        [box.apply(image) for box in self.boxes]
        return image
