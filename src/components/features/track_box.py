import random
from typing import Tuple, Union
from functools import partial

from cv2.typing import MatLike
import cv2
import numpy as np

from src.modules.utils import tuple_handler


class Box:
    def __init__(
        self, top_left: Tuple, bottom_right: Tuple, smoothness: int = 1
    ) -> None:
        self.tl = top_left
        self.br = bottom_right
        self.smoothness = smoothness
        self.config_box()
        self.config_text()
        self.history = list()
        self.count = 0

    def check(self, pos: Tuple) -> None:
        x1, y1 = self.tl
        x2, y2 = self.br

        X, Y = tuple_handler(pos, max_dim=2)

        if (x1 <= X <= x2) and (y1 <= Y <= y2):
            self.count += 1

    def get_value(self) -> int:
        self.history.append(self.count)
        if len(self.history) > self.smoothness:
            self.history.pop(0)
        self.count = 0
        return int(np.mean(self.history))

    def config_box(self, color: Tuple = 0, thickness: int = 1):
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
        self.draw_box(img=frame)
        self.add_text(img=frame, text=str(self.get_value()))


class TrackBox:
    BOXES = list()

    @classmethod
    def new(
        cls, name: str, top_left: Tuple, bottom_right: Tuple, smoothness: int = 1
    ) -> "Box":
        box = Box(top_left, bottom_right, smoothness)
        cls.BOXES.append({"name": str(name), "box": box})
        return box

    @classmethod
    def get(cls, name: str) -> Union["Box", None]:
        for box in cls.BOXES:
            if str(name) == box["name"]:
                return box["box"]
        else:
            return None

    @classmethod
    def remove(cls, name: str) -> None:
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
        __color = color
        for box in cls.BOXES:
            if not color:
                __color = tuple(random.randint(0, 255) for _ in range(3))
            box["box"].config_box(color=__color, thickness=box_thickness)
            box["box"].config_text(
                pos_adjust=text_pos_adjust,
                font_scale=font_scale,
                color=__color,
                thickness=text_thickness,
            )

    @classmethod
    def check(cls, pos: Tuple) -> None:
        [box["box"].check(pos) for box in cls.BOXES]

    @classmethod
    def show(cls, frame: MatLike) -> None:
        [box["box"].show(frame) for box in cls.BOXES]
