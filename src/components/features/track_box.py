from typing import Tuple, Union
from functools import partial
import random

import cupy as cp
import cv2

from src.modules.utils import tuple_handler


class Box:
    def __init__(
        self,
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
            top_left (Tuple): Top-left corner coordinates (x, y).
            bottom_right (Tuple): Bottom-right corner coordinates (x, y).
            smoothness (int, optional): Number of frames for smoothing. Defaults to 1.
        """
        self.tl = top_left
        self.br = bottom_right
        self.smoothness = smoothness
        self.draw_box = self.config_box(color, box_thickness)
        self.add_text = self.config_text(
            text_pos_adjust, font_scale, color, text_thickness
        )
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

    def config_box(self, color: Tuple = 0, thickness: int = 1):
        """
        Configure box drawing parameters.

        Args:
            color (Tuple, optional): Color of the box (B, G, R). Defaults to 0.
            thickness (int, optional): Thickness of the box outline. Defaults to 1.
        """
        return partial(
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
        return partial(
            cv2.putText,
            org=tuple(
                x + y for x, y in zip(self.tl, tuple_handler(pos_adjust, max_dim=2))
            ),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=int(font_scale),
            color=tuple_handler(color, max_dim=3),
            thickness=int(thickness),
        )

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
        return int(cp.mean(cp.array(self.history)))

    def show(self, frame: Union[cv2.Mat, cp.ndarray]):
        """
        Display the box and text on the given frame.

        Args:
            frame (MatLike): Icput image frame.
        """
        self.draw_box(img=frame)
        self.add_text(img=frame, text=str(self.get_value()))


class TrackBox:
    def __init__(self, **kwargs) -> None:
        self.default_config = kwargs
        self.BOXES = list()

    def new(self, **kwargs) -> "Box":
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
        kwargs.update((k, v) for k, v in self.default_config.items() if k not in kwargs)
        box = {"name": str(kwargs.pop("name")), "box": Box(**kwargs)}
        self.BOXES.append(box)
        return box

    def get(self, name: str) -> Union["Box", None]:
        """
        Get a box by name.

        Args:
            name (str): Name of the box.

        Returns:
            Union[Box, None]: The box if found, otherwise None.
        """
        for box in self.BOXES:
            if str(name) == box["name"]:
                return box["box"]
        else:
            return None

    def remove(self, name: str) -> None:
        """
        Remove a box by name.

        Args:
            name (str): Name of the box.

        Raises:
            ValueError: If the box with the specified name is not found.
        """
        box = self.get(name)
        if box:
            self.BOXES.remove({"name": name, "box": box})
        else:
            raise ValueError(f"The box named '{name}' was not found")

    def check(self, pos: Tuple) -> None:
        """
        Check if the provided position is within any of the tracked boxes.

        Args:
            pos (Tuple): Position coordinates (x, y).
        """
        [box["box"].check(pos) for box in self.BOXES]

    def show(self, frame: Union[cv2.Mat, cp.ndarray]) -> None:
        """
        Display all tracked boxes on the given frame.

        Args:
            frame (MatLike): Icput image frame.
        """
        [box["box"].show(frame) for box in self.BOXES]
