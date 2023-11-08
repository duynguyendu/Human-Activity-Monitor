import shutil

from omegaconf import DictConfig
from tqdm import tqdm
import hydra
import cv2

# Setup root directory
from rootutils import autosetup

autosetup()

from modules.data.transform import DataTransformation
from components.detectors.yolo_v8 import YoloV8
from components.classifiers.vit import ViT
from components.features import *
from modules.video import Video


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
    video = Video(path=cfg["path"])

    # Config video progress bar
    progress_bar = tqdm(
        total=video.total_frame,
        desc=video.name,
        unit=" frame",
        miniters=1,
        smoothing=0.1,
    )

    # Frame loop
    for frame in video.get_frame():
        # Frame sampling
        if progress_bar.n % cfg["sampling"] != 0:
            progress_bar.update(1)
            continue

        # Initial
        if progress_bar.n == 0:
            text_format = lambda text, pos: cv2.putText(
                frame,
                text,
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            if cfg["enable"]["person_count"]:
                human_counter = HumanCount(
                    smoothness=cfg["feature"]["person_count"]["smoothness"]
                )

            if cfg["enable"]["heatmap"]:
                heatmap = Heatmap(shape=video.size(reverse=True))

                if cfg["feature"]["heatmap"]["save_video"]:
                    heatmap.save_video(
                        save_path=f"records/{video.stem}/heatmap.mp4",
                        fps=video.fps,
                        size=video.size(),
                    )

                if cfg["feature"]["heatmap"]["save_image"]:
                    heatmap.save_image(
                        save_path=f"records/{video.stem}/heatmap.jpg",
                        size=video.size(reverse=True),
                    )

        # Detect human
        outputs = detector(frame)

        # Heatmap
        if cfg["enable"]["heatmap"]:
            for output in outputs:
                x1, y1, x2, y2 = output["box"]

                heatmap.update(
                    area=(x1, y1, x2, y2),
                    value=cfg["feature"]["heatmap"]["grow"],
                )

            # Decay
            heatmap.decay(cfg["feature"]["heatmap"]["decay"])

            # Apply
            frame, heat_layer = heatmap.apply(
                image=frame,
                blurriness=cfg["feature"]["heatmap"]["blur"],
                alpha=cfg["feature"]["heatmap"]["alpha"],
            )

        # Human count
        if cfg["enable"]["person_count"]:
            human_counter.update(value=len(outputs))
            text_format(f"Person: {human_counter.get_value()}", pos=(20, 40))

        # Human loop
        for output in outputs:
            x1, y1, x2, y2 = output["box"]

            # Human box
            if cfg["feature"]["human_box"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Human dot
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if cfg["feature"]["human_dot"]:
                cv2.circle(frame, center, 5, (225, 225, 225), -1)

            # Classify action
            if cfg["enable"]["classifier"]:
                X = transfrom(output["human"])

                result = classifier(X)

                text_format(result, pos=(x1, y1 - 5))

        cv2.imshow(f"{cfg['path']}", frame)

        progress_bar.update(1)

        if not video.delay(cfg["delay"]):
            break

    (x.release() for x in (video, heatmap))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
