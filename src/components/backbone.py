from typing import Dict, Union
import os

from rich import print
from cv2 import Mat
import numpy as np

from src.modules.utils import tuple_handler

from .features import *
from . import *


class Backbone:
    PROCESSES = {
        "detector": "Detection",
        "classifier": "Classification",
        "human_count": "Human count",
        "heatmap": "Heatmap",
        "track_box": "Track box",
    }

    def __init__(self, video: "Video", config: Dict) -> None:
        print("[bold]Summary:[/]")
        self.video = video

        print("  [bold]Process status:[/]")
        for key, process in self.PROCESSES.items():
            if key in config and config[key]:
                if key in ["detector", "classifier"]:
                    getattr(self, f"setup_{key}")(config[key], device=config["device"])
                else:
                    getattr(self, f"setup_{key}")(config[key])
                print(f"    {process}: [green]Enable[/]")
            else:
                print(f"    {process}: [red]Disable[/]")

    def __call__(self, frame: Union[np.ndarray, Mat]):
        return self.apply(frame)

    def setup_detector(self, config: Dict, device: str) -> None:
        self.detector = Detector(**config["model"], device=device)
        self.show_detected = config["show"]
        self.track = config["model"]["track"]

    def setup_classifier(self, config: Dict, device: str) -> None:
        self.classifier = Classifier(**config["classifier"], device=device)

    def setup_human_count(self, config: Dict) -> None:
        self.human_count = HumanCount(smoothness=config["smoothness"])
        if config["save"]:
            save_path = os.path.join(
                config["save"]["save_path"],
                self.video.stem,
                config["save"]["save_name"],
            )
            self.human_count.save(
                save_path=save_path + ".csv",
                interval=config["save"]["interval"],
                fps=self.video.fps,
                speed=self.video.speed,
            )
            print(f"  [bold]Save counted people to:[/] [green]{save_path}.csv[/]")

    def setup_heatmap(self, config: Dict) -> None:
        self.heatmap = Heatmap(shape=self.video.size(reverse=True), **config["layer"])

        if config["save"]:
            # Config save path
            save_path = os.path.join(
                config["save"]["save_path"],
                self.video.stem,
                config["save"]["save_name"],
            )

            # Config save resolution
            save_res = (
                tuple_handler(config["save"]["resolution"], max_dim=2)
                if config["save"]["resolution"]
                else self.video.size()
            )

            # Save video
            if config["save"]["video"]:
                self.heatmap.save_video(
                    save_path=save_path + ".mp4",
                    fps=self.video.fps,
                    size=save_res,
                )
                print(f"  [bold]Save heatmap video to:[/] [green]{save_path}.mp4[/]")

            # Save image
            if config["save"]["image"]:
                self.heatmap.save_image(
                    save_path=save_path + ".jpg",
                    size=save_res[::-1],
                )
                print(f"  [bold]Save heatmap image to:[/] [green]{save_path}.jpg[/]")

    def setup_track_box(self, config: Dict) -> None:
        self.track_box = TrackBox(**config["default"])
        [self.track_box.new(**box) for box in config["boxes"]]

    def apply(self, frame: Union[np.ndarray, Mat]):
        if not hasattr(self, "detector"):
            return frame

        outputs = self.detector(frame)

        if hasattr(self, "human_count"):
            # Update new value
            self.human_count.update(value=len(outputs))
            # Add to frame
            self.video.add_text(
                text=f"Person: {self.human_count.get_value()}",
                pos=(20, 40),
                thickness=2,
            )

        for output in outputs:
            x1, y1, x2, y2 = output["box"]

            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if self.show_detected:
                if self.show_detected["dot"]:
                    self.video.add_point(center=center, radius=5, color=(225, 225, 225))

                if self.show_detected["box"]:
                    self.video.add_box(
                        top_left=(x1, y1), bottom_right=(x2, y2), color=255, thickness=2
                    )

                if self.show_detected["score"]:
                    self.video.add_text(
                        text=f"{output['conf']:.2}", pos=(x1, y2 - 5), thickness=2
                    )

            if self.track:
                self.video.add_text(text=output["id"], pos=(x1, y1 - 5), thickness=2)

            if hasattr(self, "classifier"):
                result = self.classifier(frame[y1:y2, x1:x2])
                self.video.add_text(text=result, pos=(x1, y1 - 5), thickness=2)

            if hasattr(self, "heatmap"):
                self.heatmap.update(area=(x1, y1, x2, y2))

            if hasattr(self, "track_box"):
                self.track_box.check(pos=center)

        if hasattr(self, "heatmap"):
            self.heatmap.decay()

            frame, heat_layer = self.heatmap.apply(image=frame)

        if hasattr(self, "track_box"):
            for data in self.track_box.BOXES:
                self.video.add_box(**data["box"].box_config)
                self.video.add_text(
                    text=data["box"].get_value(), **data["box"].text_config
                )

        return frame

    def finish(self) -> None:
        if hasattr(self, "heatmap"):
            self.heatmap.release()
