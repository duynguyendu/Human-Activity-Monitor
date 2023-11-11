from typing import Tuple, Union
from functools import partial
import random

from cv2.typing import MatLike
import numpy as np
import cv2

from src.modules.utils import tuple_handler


class Box:
    def __init__(
        self, top_left: Tuple, bottom_right: Tuple, smoothness: int = 1
    ) -> None:
        """
        Initialize a box with given top-left and bottom-right coordinates.

        Args:
            top_left (Tuple): Top-left corner coordinates (x, y).
            bottom_right (Tuple): Bottom-right corner coordinates (x, y).
            smoothness (int, optional): Number of frames for smoothing. Defaults to 1.
        """
        self.tl = top_left
        self.br = bottom_right
        self.smoothness = smoothness
        self.config_box()
        self.config_text()
        self.history = list()
        self.count = 0

    def check(self, pos: Tuple) -> None:
        """
        Check if a position is within the box.

        Args:
            pos (Tuple): Position coordinates (x, y).
        """
        x1, y1 = self.tl
        x2, y2 = self.br

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
        if len(self.history) > self.smoothness:
            self.history.pop(0)
        self.count = 0
        return int(np.mean(self.history))

    def config_box(self, color: Tuple = 0, thickness: int = 1):
        """
        Configure box drawing parameters.

        Args:
            color (Tuple, optional): Color of the box (B, G, R). Defaults to 0.
            thickness (int, optional): Thickness of the box outline. Defaults to 1.
        """
        self.draw_box = partial(
            cv2.rectangle,
            pt1=tuple_handler(self.tl, max_dim=2),
            pt2=tuple_handler(self.br, max_dim=2),
            color=tuple_handler(color, max_dim=3),
            thickness=int(thickness),
        )

    def config_text(
        self,
        pos_adjust: Tuple = (0, 0),
        font_scale: int = 1,
        color: Tuple = 0,
        thickness: int = 1,
    ):
        """
        Configure text drawing parameters.

        Args:
            pos_adjust (Tuple, optional): Text position adjustment (x, y). Defaults to (0, 0).
            font_scale (int, optional): Font scale for the text. Defaults to 1.
            color (Tuple, optional): Color of the text (B, G, R). Defaults to 0.
            thickness (int, optional): Thickness of the text. Defaults to 1.
        """
        self.add_text = partial(
            cv2.putText,
            org=tuple(
                x + y for x, y in zip(self.tl, tuple_handler(pos_adjust, max_dim=2))
            ),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=int(font_scale),
            color=tuple_handler(color, max_dim=3),
            thickness=int(thickness),
        )

    def show(self, frame: MatLike):
        """
        Display the box and text on the given frame.

        Args:
            frame (MatLike): Input image frame.
        """
        self.draw_box(img=frame)
        self.add_text(img=frame, text=str(self.get_value()))


class TrackBox:
    BOXES = list()

    @classmethod
    def new(
        cls, name: str, top_left: Tuple, bottom_right: Tuple, smoothness: int = 1
    ) -> "Box":
        """
        Create a new tracked box.

        Args:
            name (str): Name of the box.
            top_left (Tuple): Top-left corner coordinates (x, y).
            bottom_right (Tuple): Bottom-right corner coordinates (x, y).
            smoothness (int, optional): Number of frames for smoothing. Defaults to 1.

        Returns:
            Box: The created box.
        """
        box = Box(top_left, bottom_right, smoothness)
        cls.BOXES.append({"name": str(name), "box": box})
        return box

    @classmethod
    def get(cls, name: str) -> Union["Box", None]:
        """
        Get a box by name.

        Args:
            name (str): Name of the box.

        Returns:
            Union[Box, None]: The box if found, otherwise None.
        """
        for box in cls.BOXES:
            if str(name) == box["name"]:
                return box["box"]
        else:
            return None

    @classmethod
    def remove(cls, name: str) -> None:
        """
        Remove a box by name.

        Args:
            name (str): Name of the box.

        Raises:
            ValueError: If the box with the specified name is not found.
        """
        box = cls.get(name)
        if box:
            cls.BOXES.remove({"name": name, "box": box})
        else:
            raise ValueError(f"The box named '{name}' was not found")

    @classmethod
    def config_boxes(
        cls,
        color: Tuple = None,
        box_thickness: int = 1,
        text_pos_adjust: Tuple = None,
        font_scale: int = 1,
        text_thickness: int = 1,
    ) -> None:
        """
        Configure drawing parameters for all tracked boxes.

        Args:
            color (Tuple, optional): Color of the boxes (B, G, R). If not provided, random colors are assigned.
            box_thickness (int, optional): Thickness of the box outlines. Defaults to 1.
            text_pos_adjust (Tuple, optional): Text position adjustment for all boxes (x, y). Defaults to None.
            font_scale (int, optional): Font scale for the text. Defaults to 1.
            text_thickness (int, optional): Thickness of the text. Defaults to 1.
        """
        default_color = color

        for box in cls.BOXES:
            if not color:
                default_color = tuple(random.randint(0, 255) for _ in range(3))
            box["box"].config_box(color=default_color, thickness=box_thickness)
            box["box"].config_text(
                pos_adjust=text_pos_adjust,
                font_scale=font_scale,
                color=default_color,
                thickness=text_thickness,
            )

    @classmethod
    def check(cls, pos: Tuple) -> None:
        """
        Check if the provided position is within any of the tracked boxes.

        Args:
            pos (Tuple): Position coordinates (x, y).
        """
        [box["box"].check(pos) for box in cls.BOXES]

    @classmethod
    def show(cls, frame: MatLike) -> None:
        """
        Display all tracked boxes on the given frame.

        Args:
            frame (MatLike): Input image frame.
        """
        [box["box"].show(frame) for box in cls.BOXES]
