from omegaconf import DictConfig
import shutil
import hydra
import cv2

# Setup root directory
from rootutils import autosetup

autosetup()

from modules.data.transform import DataTransformation
from components.detectors.yolo_v8 import YoloV8
from components.classifiers.vit import ViT


@hydra.main(config_path="../configs", config_name="run", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Remove the hydra outputs since we already have lightning logs
    shutil.rmtree("outputs")

    detector = YoloV8(conf=0.25, iou=0.3)

    classifier = ViT("logs/new/version_0/checkpoints/epoch=56-step=3192.ckpt")

    cap = cv2.VideoCapture(cfg["path"])

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        outputs = detector(frame)

        for output in outputs:
            X = output["human"]
            X = DataTransformation(224).TOPIL(X).unsqueeze(0).to("cuda")

            result = classifier(X)

            x1, y1, x2, y2 = output["box"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), 225, 2)

            cv2.putText(
                frame, result, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2
            )

            # center = ((x1 + x2) // 2, (y1 + y2) // 2)
            # cv2.circle(frame, center, 5, 225, -1)
            # cv2.putText(frame, str(idx), center, cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit()

        # cv2.destroyAllWindows()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
