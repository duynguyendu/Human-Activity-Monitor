from typing import Dict, Union
import os

from rich import print
from cv2 import Mat
import numpy as np

from src.modules.utils import tuple_handler

from .features import *
from . import *


class Backbone:
    def __init__(self, video: "Video", config: Dict) -> None:
        """
        Initialize the Backbone object.

        Args:
            video (Video): An object representing the video input.
            config (Dict): Configuration settings for different processes.
        """
        print("[bold]Summary:[/]")
        self.video = video

        # Process status
        self.status = {
            process: False
            for process in [
                "detector",
                "classifier",
                "human_count",
                "heatmap",
                "track_box",
            ]
        }

        # Setup each process
        for process in self.status:
            if config.get(process, False) or config["features"].get(process, False):
                args = (
                    [config["features"][process]]
                    if process not in ["detector", "classifier"]
                    else [config[process], config["device"]]
                )
                getattr(self, f"setup_{process}")(*args)

    def __call__(self, frame: Union[np.ndarray, Mat]) -> Union[np.ndarray, Mat]:
        """
        Applies the configured processes to the input frame.

        Args:
            frame (Union[np.ndarray, Mat]): The input frame.

        Returns:
            Union[np.ndarray, Mat]: The output frame.
        """
        return self.apply(frame)

    def setup_detector(self, config: Dict, device: str) -> None:
        """
        Sets up the detector module with the specified configuration.

        Args:
            config (Dict): Configuration settings for the detector.
            device (str): The device on which the detector will run.

        Returns:
            None
        """
        self.detector = Detector(**config["model"], device=device)
        self.show_detected = config["show"]
        self.track = config["model"]["track"]

    def setup_classifier(self, config: Dict, device: str) -> None:
        """
        Sets up the classifier module with the specified configuration.

        Args:
            config (Dict): Configuration settings for the classifier.
            device (str): The device on which the classifier will run.

        Returns:
            None
        """
        self.classifier = Classifier(**config["model"], device=device)
        self.show_classified = config["show"]

    def setup_human_count(self, config: Dict) -> None:
        """
        Sets up the human count module with the specified configuration.

        Args:
            config (Dict): Configuration settings for human count.

        Returns:
            None
        """
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
        """
        Sets up the heatmap module with the specified configuration.

        Args:
            config (Dict): Configuration settings for the heatmap.

        Returns:
            None
        """
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
        """
        Sets up the track box module with the specified configuration.

        Args:
            config (Dict): Configuration settings for the track box.

        Returns:
            None
        """
        self.track_box = TrackBox(**config["default"])
        [self.track_box.new(**box) for box in config["boxes"]]

    def apply(self, frame: Union[np.ndarray, Mat]) -> Union[np.ndarray, Mat]:
        """
        Applies the configured processes to the input frame.

        Args:
            frame (Union[np.ndarray, Mat]): The input frame.

        Returns:
            Union[np.ndarray, Mat]: The output frame.
        """

        # Skip all of the process if detector is not specified
        if not (hasattr(self, "detector") and self.status["detector"]):
            return frame

        # Get detector output
        boxes = self.detector(frame)

        # Lambda function for dynamic color apply
        dynamic_color = lambda x: (0, x * 400, ((1 - x) * 400))

        # Human count
        if hasattr(self, "human_count"):
            # Update new value
            self.human_count.update(value=len(boxes))
            # Add to frame
            self.video.add_text(
                text=f"Person: {self.human_count.get_value()}",
                pos=(20, 40),
                thickness=2,
            )

        # Loop through the boxes
        for detect_output in boxes:
            # xyxy location
            x1, y1, x2, y2 = detect_output["box"]

            # Center point
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Check detector show options
            if self.show_detected:
                # Apply dynamic color
                color = (
                    dynamic_color(detect_output["score"])
                    if self.show_detected["dynamic_color"]
                    else 255
                )

                # Show dot
                if self.show_detected["dot"]:
                    self.video.add_point(center=center, radius=5, color=color)

                # Show box
                if self.show_detected["box"]:
                    self.video.add_box(
                        top_left=(x1, y1),
                        bottom_right=(x2, y2),
                        color=color,
                        thickness=2,
                    )

                # Show score
                if self.show_detected["score"]:
                    self.video.add_text(
                        text=f"{detect_output['score']:.2}",
                        pos=(x1, y2 - 5),
                        color=color,
                        thickness=2,
                    )

            # Show id it track
            if self.track:
                self.video.add_text(
                    text=detect_output["id"], pos=(x1, y1 - 5), thickness=2
                )

            # Classification
            if (
                hasattr(self, "classifier")
                and self.show_classified
                and self.status["classifier"]
            ):
                # Add box margin
                box_margin = 10
                human_box = frame[
                    y1 - box_margin : y2 + box_margin, x1 - box_margin : x2 + box_margin
                ]
                # Get model output
                classify_output = self.classifier(human_box)
                # Format result
                classify_result = ""
                if self.show_classified["text"]:
                    classify_result += classify_output["label"]
                if self.show_classified["score"]:
                    classify_result += f' ({classify_output["score"]:.2})'
                # Add to frame, color based on score
                self.video.add_text(
                    text=classify_result,
                    pos=(x1, y1 - 5),
                    color=(
                        dynamic_color(classify_output["score"])
                        if self.show_classified["dynamic_color"]
                        else 255
                    ),
                    thickness=2,
                )

            # Update heatmap
            if hasattr(self, "heatmap") and self.status["heatmap"]:
                self.heatmap.update(area=(x1, y1, x2, y2))

            # Check for track box
            if hasattr(self, "track_box") and self.status["track_box"]:
                self.track_box.check(pos=center)

        # Apply heatmap
        if hasattr(self, "heatmap") and self.status["heatmap"]:
            self.heatmap.decay()
            frame, heat_layer = self.heatmap.apply(image=frame)

        # Add track box to frame
        if hasattr(self, "track_box") and self.status["track_box"]:
            for data in self.track_box.BOXES:
                self.video.add_box(**data["box"].box_config)
                self.video.add_text(
                    text=data["box"].get_value(), **data["box"].text_config
                )

        return frame

    def finish(self) -> None:
        """
        Call finish process. Releases resources associated.

        Returns:
            None
        """
        if hasattr(self, "heatmap"):
            self.heatmap.release()
