from functools import cached_property
from typing import Dict, Union
from copy import deepcopy
from queue import Queue
import threading
import os

from rich import print
from cv2 import Mat
import numpy as np
import cv2

from src.modules.utils import tuple_handler

from .features import *
from . import *


class Backbone:
    def __init__(
        self,
        video: "Video",
        mask: bool = False,
        thread: bool = False,
        background: bool = False,
        config: Dict = None,
    ) -> None:
        """
        Initialize the Backbone object.

        Args:
            video (Video): An object representing the video input.
            mask (bool): Apply the result to a mask instead. Default to False.
            thread (bool): Run process on a separate thread. Default to False.
            background (bool): Allow process to run on background. Default to False.
            config (Dict): Configuration settings for different processes.
        """
        self.video = video
        self.mask = mask
        self.thread = thread
        self.background = background
        self.queue = Queue()

        # Process status:
        #   True by default
        self.status = {"detector": True, "human_count": True}
        #   False by default
        self.status.update(
            {
                process: False
                for process in [
                    "classifier",
                    "heatmap",
                    "track_box",
                ]
            }
        )

        print("[bold]Summary:[/]")

        # Setup each process
        for process in self.status:
            if config.get(process, False) or config["features"].get(process, False):
                args = (
                    [config["features"][process]]
                    if process not in ["detector", "classifier"]
                    else [config[process], config["device"]]
                )
                getattr(self, f"_setup_{process}")(*args)

    def __call__(self, frame: Union[np.ndarray, Mat]) -> Union[np.ndarray, Mat]:
        """
        Applies the configured processes to the input frame.

        Args:
            frame (Union[np.ndarray, Mat]): The input frame.

        Returns:
            Union[np.ndarray, Mat]: The output frame.
        """
        return self.process(frame)

    def _setup_detector(self, config: Dict, device: str) -> None:
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

    def _setup_classifier(self, config: Dict, device: str) -> None:
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

    def _setup_human_count(self, config: Dict) -> None:
        """
        Sets up the human count module with the specified configuration.

        Args:
            config (Dict): Configuration settings for human count.

        Returns:
            None
        """
        self.human_count = HumanCount(smoothness=config["smoothness"])
        self.human_count_position = config["position"]

        if config["save"]:
            save_path = os.path.join(
                config["save"]["save_path"],
                self.video.stem,
                config["save"]["save_name"],
            )
            self.human_count.config_save(
                save_path=save_path + ".csv",
                interval=config["save"]["interval"],
                fps=int(self.video.fps / self.video.subsampling),
                speed=self.video.speed,
            )
            print(f"  [bold]Save counted people to:[/] [green]{save_path}.csv[/]")

    def _setup_heatmap(self, config: Dict) -> None:
        """
        Sets up the heatmap module with the specified configuration.

        Args:
            config (Dict): Configuration settings for the heatmap.

        Returns:
            None
        """
        self.heatmap = Heatmap(shape=self.video.size(reverse=True), **config["layer"])
        self.heatmap_opacity = config["opacity"]

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
                    fps=self.video.fps / self.video.subsampling,
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

    def _setup_track_box(self, config: Dict) -> None:
        """
        Sets up the track box module with the specified configuration.

        Args:
            config (Dict): Configuration settings for the track box.

        Returns:
            None
        """
        self.track_box = TrackBox(**config["default"])
        [self.track_box.new(**box) for box in config["boxes"]]

    @cached_property
    def __new_mask(self) -> np.ndarray:
        """
        Return new video mask

        Returns:
            np.ndarray: new mask
        """
        return np.zeros((*self.video.size(reverse=True), 3), dtype=np.uint8)

    def __process_is_activate(self, name: str, background: bool = False) -> bool:
        """Check if a process is activate"""
        return hasattr(self, name) and (
            self.status[name] or (self.background if background else False)
        )

    def __threaded_process(func):
        """Move process to a separate thread"""

        def wrapper(self, frame):
            # Check if using Thread
            if not self.thread:
                return func(self, frame)

            # Turn mask on when using Thread
            elif not self.mask:
                self.mask = True

            # Only spawn Thread on first run or Thread is free
            if (
                not hasattr(self, "current_process")
                or not self.current_process.is_alive()
            ):
                # Spam a new thread
                self.current_process = threading.Thread(
                    target=func, args=(self, frame), daemon=True
                )

                # Start running the thread
                self.current_process.start()

        return wrapper

    @__threaded_process
    def process(self, frame: Union[np.ndarray, Mat]) -> None:
        """
        Process the input frame.

        Args:
            frame (Union[np.ndarray, Mat]): The input frame.

        Returns:
            None: Result is stored in a safe thread.
        """

        # Check mask option
        mask = deepcopy(self.__new_mask if self.mask else frame)

        # Skip all of the process if detector is not specified
        if self.__process_is_activate("detector", background=True):
            # Get detector output
            boxes = self.detector(frame)

            # Lambda function for dynamic color apply
            dynamic_color = lambda x: (0, x * 400, ((1 - x) * 400))

            # Human count
            if hasattr(self, "human_count"):
                # Update new value
                self.human_count.update(value=len(boxes))
                # Add to frame
                cv2.putText(
                    img=mask,
                    text=f"Person: {self.human_count.get_value()}",
                    org=self.human_count_position,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=tuple_handler(255, max_dim=3),
                    thickness=2,
                )

            # Loop through the boxes
            for detect_output in boxes:
                # xyxy location
                x1, y1, x2, y2 = detect_output["box"]

                # Center point
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Check detector show options
                if self.__process_is_activate("detector") and self.show_detected:
                    # Apply dynamic color
                    color = (
                        dynamic_color(detect_output["score"])
                        if self.show_detected["dynamic_color"]
                        else 255
                    )

                    # Show dot
                    if self.show_detected["dot"]:
                        cv2.circle(
                            img=mask,
                            center=center,
                            radius=5,
                            color=color,
                            thickness=-1,
                        )

                    # Show box
                    if self.show_detected["box"]:
                        cv2.rectangle(
                            img=mask,
                            pt1=(x1, y1),
                            pt2=(x2, y2),
                            color=color,
                            thickness=2,
                        )

                    # Show score
                    if self.show_detected["score"]:
                        cv2.putText(
                            img=mask,
                            text=f"{detect_output['score']:.2}",
                            org=(x1, y2 - 5),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=color,
                            thickness=2,
                        )

                # Show id it track
                if self.track:
                    cv2.putText(
                        img=mask,
                        text=detect_output["id"],
                        org=(x1, y1 - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=tuple_handler(255, max_dim=2),
                        thickness=2,
                    )

                # Classification
                if self.__process_is_activate("classifier") and self.show_classified:
                    # Add box margin
                    box_margin = 10
                    human_box = frame[
                        y1 - box_margin : y2 + box_margin,
                        x1 - box_margin : x2 + box_margin,
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
                    cv2.putText(
                        img=mask,
                        text=classify_result,
                        org=(x1, y1 - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(
                            dynamic_color(classify_output["score"])
                            if self.show_classified["dynamic_color"]
                            else 255
                        ),
                        thickness=2,
                    )

                # Update heatmap
                if self.__process_is_activate("heatmap", background=True):
                    self.heatmap.check(area=(x1, y1, x2, y2))

                # Check for track box
                if self.__process_is_activate("track_box"):
                    self.track_box.check(pos=center)

            # Apply heatmap
            if self.__process_is_activate("heatmap", background=True):
                self.heatmap.update()

            # Add track box to frame
            if hasattr(self, "track_box") and self.status["track_box"]:
                for data in self.track_box.BOXES:
                    cv2.rectangle(mask, *data["box"].box_config.values())
                    cv2.putText(
                        mask,
                        str(data["box"].get_value()),
                        list(data["box"].text_config.values())[0],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        *list(data["box"].text_config.values())[1:],
                    )

        # Put result to a safe thread
        self.queue.put(mask)

    def apply(self, frame: Union[np.ndarray, Mat]) -> Union[np.ndarray, Mat]:
        """
        Apply process result to the given frame.

        Args:
            frame (Union[np.ndarray, Mat]): Input frame to which the overlay will be applied.

        Returns:
            Union[np.ndarray, Mat]: Frame with overlay applied.
        """

        # Check if any processes are completed
        if not self.queue.empty():
            self.overlay = self.queue.get()
            self.filter = cv2.cvtColor(self.overlay, cv2.COLOR_BGR2GRAY) != 0

        # Return on result is empty and not mask
        elif not self.mask:
            return frame

        # Enable heatmap
        if self.__process_is_activate("heatmap") and hasattr(self.heatmap, "heatmap"):
            cv2.addWeighted(
                src1=self.heatmap.get(),
                alpha=self.heatmap_opacity,
                src2=frame,
                beta=1 - self.heatmap_opacity,
                gamma=0,
                dst=frame,
            )

        # Return overlay when not using mask
        if not self.mask:
            return self.overlay

        # Check if first run
        if hasattr(self, "overlay"):
            frame[self.filter] = self.overlay[self.filter]

        return frame

    def finish(self) -> None:
        """
        Call finish process. Releases resources associated.

        Returns:
            None
        """
        if hasattr(self, "heatmap"):
            self.heatmap.release()
