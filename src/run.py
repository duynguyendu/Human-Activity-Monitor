import shutil

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
    # Transform
    transfrom = DataTransformation.TOPIL(image_size=cfg["image_size"])

    # Detector
    detector = YoloV8(**cfg["detector"], size=cfg["image_size"], device=cfg["device"])

    # Classifier
    classifier = ViT(**cfg["classifier"])

    # Open video
    cap = cv2.VideoCapture(cfg["path"])

    # Heatmap
    heatmap = Heatmap(
        blurriness=cfg["utils"]["heatmap"]["blur"],
        alpha=cfg["utils"]["heatmap"]["alpha"],
    )

    heatmap.layer = np.zeros(
        shape=(
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        ),
        dtype=np.uint32,
    )

    # Video progess
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    progress_bar = tqdm(
        total=total_frames,
        desc="Frame",
        miniters=1,
        smoothing=0.1,
    )

    while cap.isOpened():
        ret, frame = cap.read()

        if progress_bar.n % cfg["sampling"] != 0:
            progress_bar.update(1)
            continue

        if not ret:
            break

        # Detect human
        outputs = detector(frame)

        # Human loop
        for output in outputs:
            x1, y1, x2, y2 = output["box"]

            # Human box
            cv2.rectangle(frame, (x1, y1), (x2, y2), 255, 2)

            # Update heatmap
            heatmap.update(area=(x1, y1, x2, y2), value=cfg["utils"]["heatmap"]["grow"])

            # Classify action
            X = transfrom(output["human"])

            result = classifier(X)

            cv2.putText(
                frame, result, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2
            )

            # center = ((x1 + x2) // 2, (y1 + y2) // 2)
            # cv2.circle(frame, center, 5, 225, -1)
            # cv2.putText(frame, str(idx), center, cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

        # Apply heatmap
        frame = heatmap.apply(frame)

        cv2.imshow("frame", frame)

        if cv2.waitKey(cfg["delay"]) & 0xFF == ord("q"):
            exit()

        progress_bar.update(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
