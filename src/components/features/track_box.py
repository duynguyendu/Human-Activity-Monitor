from typing import Tuple, Union

import cupy as cp

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
        self.xyxy = (*top_left, *bottom_right)
        self.smoothness = smoothness
        self.history = list()
        self.count = 0
        self.box_config = {
            "top_left": top_left,
            "bottom_right": bottom_right,
            "color": color,
            "thickness": box_thickness,
        }
        self.text_config = {
            "pos": tuple(x + y for x, y in zip(top_left, text_pos_adjust)),
            "font_scale": font_scale,
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
        if len(self.history) > self.smoothness:
            self.history.pop(0)
        self.count = 0
        return int(cp.mean(cp.array(self.history)))


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
