import shutil
import os

from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np
import hydra
import cv2

# Setup root directory
from rootutils import autosetup

autosetup()

from modules.data.transform import DataTransformation
from components.detectors.yolo_v8 import YoloV8
from components.classifiers.vit import ViT
from components.utils.heatmap import Heatmap


@hydra.main(config_path="../configs", config_name="run", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Remove the hydra outputs since we already have lightning logs
    shutil.rmtree("outputs")

    # Define main componets
    # Detector
    detector = YoloV8(**cfg["detector"], size=cfg["image_size"], device=cfg["device"])

    # Transform
    transfrom = DataTransformation.TOPIL(image_size=cfg["image_size"])

    # Classifier
    classifier = ViT(**cfg["classifier"])

    # Open video
    if not os.path.exists(cfg["path"]):
        raise FileNotFoundError(
            f"Cannot locate {cfg['path']}. Use full path or check again."
        )
    cap = cv2.VideoCapture(cfg["path"])

    # Video progess
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Config video progress bar
    progress_bar = tqdm(
        total=total_frames,
        desc=f"{cfg['path'].split('/')[-1]}",
        unit=" frame",
        miniters=1,
        smoothing=1,
    )

    # Frame loop
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Frame sampling
        if progress_bar.n % cfg["sampling"] != 0:
            progress_bar.update(1)
            continue

        # Initial
        if progress_bar.n == 0:
            pause = False
            counter = list()

            if cfg["enable"]["heatmap"]:
                Heatmap.LAYER = Heatmap.new_layer_from_image(frame)

        # Detect human
        outputs = detector(frame)

        # Human counter smoothness
        counter.append(len(outputs))
        if len(counter) > cfg["feature"]["person_count"]["smoothness"]:
            counter.pop(0)

            # center = ((x1 + x2) // 2, (y1 + y2) // 2)
            # cv2.circle(frame, center, 5, 225, -1)
            # cv2.putText(frame, str(idx), center, cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

        # Heatmap
        if cfg["enable"]["heatmap"]:
            for output in outputs:
                x1, y1, x2, y2 = output["box"]

                # Update heatmap
                Heatmap.update(
                    area=(x1, y1, x2, y2),
                    value=cfg["feature"]["heatmap"]["grow"],
                )

            # Decay
            Heatmap.decay(cfg["feature"]["heatmap"]["decay"])

            # Apply
            frame, heatmap = Heatmap.apply(
                image=frame,
                blurriness=cfg["feature"]["heatmap"]["blur"],
                alpha=cfg["feature"]["heatmap"]["alpha"],
            )

        # Human loop
        for output in outputs:
            x1, y1, x2, y2 = output["box"]

            # Human box
            if cfg["feature"]["human_box"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Classify action
            if cfg["enable"]["classifier"]:
                X = transfrom(output["human"])

                result = classifier(X)

                cv2.putText(
                    frame,
                    result,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

        # Show total number of people
        if cfg["enable"]["person_count"]:
            cv2.putText(
                frame,
                f"People: {int(np.mean(counter))}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        cv2.imshow(f"{cfg['path']}", frame)
        progress_bar.update(1)

        key = cv2.waitKey(cfg["delay"] if not pause else 0) & 0xFF
        if key == ord("q"):
            exit()
        if key == ord("p"):
            pause = True
        if key == ord("r"):
            pause = False
        if key == ord("c"):
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
