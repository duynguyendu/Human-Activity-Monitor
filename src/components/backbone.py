from functools import cached_property
from typing import Dict, Union
from datetime import datetime
from collections import deque
from copy import deepcopy
from queue import Queue
import threading
import os

from rich import print
from cv2 import Mat
import numpy as np
import cv2

from .features import *
from . import *


class Backbone:
    def __init__(
        self,
        video: "Video",
        mask: bool = False,
        thread: bool = False,
        background: bool = False,
        show_latency: bool = False,
        save: Union[Dict, bool] = None,
        process_config: Union[Dict, bool] = None,
    ) -> None:
        """
        Initialize the Backbone object.

        Args:
            video (Video): An object representing the video input.
            mask (bool): Apply the result to a mask instead. Default to False.
            thread (bool): Run process on a separate thread. Default to False.
            background (bool): Allow process to run on background. Default to False.
            show_latency (bool): Display backbone latency. Default to False.
            save (Dict or bool): Configuration for save result.
            process_config (Dict or bool): Configuration settings for various processes.
        """
        self.video = video
        self.mask = True if thread else mask
        self.thread = thread
        self.background = background
        self.__setup_save(config=save)
        self.__setup_latency(config=show_latency)
        self.queue = Queue()

        # Process status:
        #   True by default
        self.status = {"detector": True, "human_count": True, "classifier": True, "tracker": True}
        #   False by default
        self.status.update(
            {process: False for process in ["heatmap", "track_box"]}
        )

        # Setup each process
        for process in self.status:
            if process_config.get(process, False) or process_config["features"].get(
                process, False
            ):
                args = (
                    [process_config["features"][process]]
                    if process not in ["detector", "classifier", "tracker"]
                    else [process_config[process], process_config["device"]]
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

    def __setup_save(self, config: Union[Dict, bool]) -> None:
        """
        Setup for save process result.

        Args:
            config (Dict or bool): A dictionary of configurations. False for disable.
        """

        # Disable if config is not provided
        if not config:
            return

        # Set save path
        self.save_path = os.path.join(
            config["path"],
            (
                datetime.now().strftime("%d-%m-%Y")
                if self.video.is_camera
                else self.video.stem
            ),
        )

        # Create destination folder
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        # Create writer
        self.writer = Writer(self.save_path, config["interval"], self.video.is_camera)

        # Logging
        print(f"[INFO] [bold]Save process result to:[/] [green]{self.save_path}[/]")

    def __setup_latency(self, config: bool) -> None:
        """Setup for tracking latency"""
        self.latency_history = deque([], maxlen=self.video.fps)
        self.process_index = {"prev": 0, "curr": 0}
        self.show_latency = config

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

        config["model"]["weight"] = "weights/phone.pt"
        config["model"]["conf"] = 0.4

        self.phone_detector = Detector(**config["model"], device=device)

    def _setup_classifier(self, config: Dict, device: str) -> None:
        """
        Sets up the classifier module with the specified configuration.

        Args:
            config (Dict): Configuration settings for the classifier.
            device (str): The device on which the classifier will run.

        Returns:
            None
        """
        self.classifier = Detector(**config["model"], device=device)
        self.show_classified = config["show"]

        if hasattr(self, "save_path") and config["save"]:
            self.writer.new(name="tracker", type="csv", features="output")

    def _setup_tracker(self, config: Dict, device: str) -> None:
        """
        Sets up the tracker module with the specified configuration.

        Args:
            config (Dict): Configuration settings.
            device (str): The device on which this will run.
        """
        if hasattr(self, "detector"):
            self.tracker = Tracker(
                **config,
                det_conf=self.detector.config["conf"],
                det_iou=self.detector.config["iou"],
                device=device,
            )

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

        if hasattr(self, "save_path") and config["save"]:
            self.writer.new(name="human_count", type="csv", features="value")

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

        if hasattr(self, "save_path") and config["save"]:
            # Save video
            if config["save"]["video"]:
                self.heatmap.save_video(
                    save_path=os.path.join(self.save_path, "heatmap.mp4"),
                    fps=self.video.fps / self.video.subsampling,
                    size=self.video.size(),
                )
            # Save image
            if config["save"]["image"]:
                self.heatmap.save_image(
                    save_path=os.path.join(self.save_path, "heatmap.jpg"),
                    size=self.video.size(reverse=True),
                )

    def _setup_track_box(self, config: Dict) -> None:
        """
        Sets up the track box module with the specified configuration.

        Args:
            config (Dict): Configuration settings for the track box.

        Returns:
            None
        """
        self.track_box = TrackBox(
            default_config=config["default"], boxes=config["boxes"]
        )
        if hasattr(self, "save_path") and config["save"]:
            self.writer.new(
                name="track_box",
                type="csv",
                features=[box.name for box in self.track_box.boxes],
            )

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

            # Spam a new thread
            self.current_process = threading.Thread(
                target=func, args=(self, frame), daemon=False
            )

            # Start running the thread
            self.current_process.start()

        return wrapper

    def is_free(self) -> bool:
        """
        Check status of backbone process

        Returns:
            bool: True if there are no processes else False
        """
        return (
            not hasattr(self, "current_process") or not self.current_process.is_alive()
        )

    def update_progress(self) -> None:
        """Update process index"""
        self.process_index["prev"] = self.process_index["curr"]
        self.process_index["curr"] = self.video.current_progress

    def update_latency(self) -> None:
        """Update latency history"""
        self.latency_history.append(
            (self.video.current_progress - self.process_index["prev"]) / self.video.fps
        )

    def get_latency(self) -> float:
        """
        Get current backbone latency

        Returns:
            float: Average latency the last 1 second
        """
        return np.mean(self.latency_history)

    def dynamic_color(self, value: int) -> tuple:
        """
        Generate color based on input value.

        Args:
            value (int): Score used to create color.

        Returns:
            tuple: Corresponding color.
        """
        return (0, value * 400, ((1 - value) * 400))

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

        # Current video process
        video_progress = int(self.video.current_progress / self.video.fps)

        # Skip all of the process if detector is not specified
        if self.__process_is_activate("detector", background=True):
            # Get detector output
            boxes = self.detector(frame)

            # Human count
            if hasattr(self, "human_count"):
                # Update new value
                self.human_count.update(value=len(boxes))

                # Save output
                if self.writer.has("human_count"):
                    self.writer.save(
                        name="human_count",
                        contents=self.human_count.get_value(),
                        progress=video_progress,
                    )

                # Add to frame
                cv2.putText(
                    img=mask,
                    text=f"Person: {self.human_count.get_value()}",
                    org=self.human_count_position,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 255, 0),
                    thickness=2,
                )

            # Tracking
            if self.__process_is_activate("tracker"):
                # Get updated result
                boxes = self.tracker.update(dets=boxes, image=frame)

            # Initialize boxes data
            boxes_data = []

            # Loop through the boxes
            for box in boxes:
                data = {}

                # xyxy location
                x1, y1, x2, y2 = map(int, box[:4])

                # Center point
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Check detector show options
                if self.__process_is_activate("detector") and self.show_detected:
                    # Apply dynamic color
                    color = (
                        self.dynamic_color(box[4])
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
                            text=f"{box[4]:.2}",
                            org=(x1, y1 + 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=color,
                            thickness=2,
                        )

                # Show id it track
                if all(map(self.__process_is_activate, ["detector", "tracker"])):
                    cv2.putText(
                        img=mask,
                        text=str(int(box[5])),
                        org=(x1, y2 - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(255, 255, 0),
                        thickness=2,
                    )

                # Phone
                if hasattr(self, "phone_detector"):
                    data["action"] = "working"

                    phone_result = self.phone_detector(frame)

                    for phone in phone_result:
                        # xyxy location
                        p_x1, p_y1, p_x2, p_y2 = map(int, phone[:4])

                        # Center point
                        p_x_center = (p_x1 + p_x2) // 2
                        p_y_center = (p_y1 + p_y2) // 2

                        if (x1 <= p_x_center <= x2) and (y1 <= p_y_center <= y2):
                            data["action"] = "phone"
                            break

                # Classification
                if (
                    self.__process_is_activate("detector")
                    and self.__process_is_activate("classifier")
                    and self.show_classified
                ):
                    # Get model output
                    classify_output = self.classifier(frame[y1:y2, x1:x2])

                    classify_label = ["other", "uniform"][np.argmax(classify_output)]

                    classify_score = classify_output[np.argmax(classify_output)]

                    data["uniform"] = classify_label == "uniform"

                    # Format result
                    classify_result = ""
                    if self.show_classified["text"]:
                        classify_result += classify_label

                    if self.show_classified["score"]:
                        classify_result += f" ({classify_score:.2})"

                    # Add to frame, color based on score
                    cv2.putText(
                        img=mask,
                        text=classify_result,
                        org=(x1, y1 - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(
                            self.dynamic_color(classify_score)
                            if self.show_classified["dynamic_color"]
                            else 255
                        ),
                        thickness=2,
                    )

                # Update tracker
                if self.__process_is_activate("tracker"):
                    boxes_data.append({"id": int(box[5]), "data": data})

                # Update heatmap
                if self.__process_is_activate("heatmap", background=True):
                    self.heatmap.check(area=(x1, y1, x2, y2))

                # Check for track box
                if self.__process_is_activate("track_box"):
                    self.track_box.check(pos=center)

            # Save classifier output
            if self.writer.has("tracker") and self.__process_is_activate("tracker"):
                self.writer.save(
                    name="tracker", contents=str(boxes_data), progress=video_progress
                )

            # Apply heatmap
            if self.__process_is_activate("heatmap", background=True):
                self.heatmap.update()

            # Add track box to frame
            if hasattr(self, "track_box") and self.status["track_box"]:
                self.track_box.update()
                self.track_box.apply(mask)

                # Save output
                if self.writer.has("track_box"):
                    self.writer.save(
                        name="track_box",
                        contents=[box.get_value() for box in self.track_box.boxes],
                        progress=video_progress,
                    )

        # Put result to a safe thread
        self.queue.put(mask)

        # Update current process
        self.update_progress()

    def apply(self, frame: Union[np.ndarray, Mat]) -> Union[np.ndarray, Mat]:
        """
        Apply process result to the given frame.

        Args:
            frame (Union[np.ndarray, Mat]): Input frame to which the overlay will be applied.

        Returns:
            Union[np.ndarray, Mat]: Frame with overlay applied.
        """

        # Update latency history
        self.update_latency()

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
                src2=frame if self.mask else self.overlay,
                beta=1 - self.heatmap_opacity,
                gamma=0,
                dst=frame if self.mask else self.overlay,
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
