import shutil
import os

from omegaconf import DictConfig
from rich import print
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


@hydra.main(config_path="../configs/run", config_name="run", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Remove the hydra outputs since we already have lightning logs
    shutil.rmtree("outputs")

    # Define main componets
    # Detector
    DETECTOR = YoloV8(**cfg["detector"]["model"], device=cfg["device"])

    if cfg["classifier"]:
        # Transform
        TRANSFORM = DataTransformation.TOPIL(image_size=cfg["classifier"]["image_size"])

        # Classifier
        CLASSIFIER = ViT(
            checkpoint=cfg["classifier"]["checkpoint"], device=cfg["device"]
        )

    # Load video
    VIDEO = Video(path=cfg["video"]["path"])

    # Track box
    # Boxes initialize
    [TrackBox.new(**box) for box in cfg["features"]["track_box"]["boxes"]]
    # Configure boxes
    TrackBox.config_boxes(**cfg["features"]["track_box"]["config"])

    # Config video progress bar
    progress_bar = tqdm(
        total=VIDEO.total_frame,
        desc=f"  {VIDEO.name}",
        unit=" frame",
        miniters=1,
        smoothing=0.1,
        delay=0.5,
    )

    # Frame loop
    for frame in VIDEO:
        # Frame sampling
        if (progress_bar.n % max(1, cfg["video"]["speed"])) != 0:
            progress_bar.update(1)
            continue

        # First run setup
        if progress_bar.n == 0:
            print("[bold]Initialize:[/]")

            # Initialize video writer
            writer_cfg = cfg["video"]["save"]
            if writer_cfg:
                save_path = os.path.join(
                    writer_cfg["save_path"],
                    VIDEO.stem,
                    writer_cfg["save_name"] + ".mp4",
                )
                writer = VIDEO.writer(save_path=save_path)
                print(f"  [bold]Saving output to:[/] [green]{save_path}[/]")

            # Initialize person count
            human_count_cfg = cfg["features"]["human_count"]
            if human_count_cfg:
                human_counter = HumanCount(smoothness=human_count_cfg["smoothness"])
                human_counter.save_config(
                    save_path=os.path.join(
                        human_count_cfg["save"]["save_path"],
                        VIDEO.stem,
                        human_count_cfg["save"]["save_name"] + ".csv",
                    ),
                    interval=human_count_cfg["save"]["interval"],
                    speed=cfg["video"]["speed"],
                )

            # Initialize heatmap
            heatmap_cfg = cfg["features"]["heatmap"]
            if heatmap_cfg:
                heatmap = Heatmap(shape=VIDEO.size(reverse=True))

                # Heatmap config save
                hm_save_path = os.path.join(
                    heatmap_cfg["save"]["save_path"],
                    VIDEO.stem,
                    heatmap_cfg["save"]["save_name"],
                )
                # Check video save
                if heatmap_cfg["save"]["video"]:
                    heatmap.save_video(
                        save_path=hm_save_path + ".mp4",
                        fps=VIDEO.fps,
                        size=VIDEO.size(),
                    )
                    print(
                        f"  [bold]Saving heatmap video to:[/] [green]{hm_save_path}.mp4[/]"
                    )
                # Check image save
                if heatmap_cfg["save"]["image"]:
                    heatmap.save_image(
                        save_path=hm_save_path + ".jpg",
                        size=VIDEO.size(reverse=True),
                    )
                    print(
                        f"  [bold]Saving heatmap image to:[/] [green]{hm_save_path}.jpg[/]"
                    )
            print(f"[bold]Video progress:[/]")

        # Detect human
        outputs = DETECTOR(frame)

        # Heatmap
        if heatmap_cfg:
            for output in outputs:
                x1, y1, x2, y2 = output["box"]

                heatmap.update(
                    area=(x1, y1, x2, y2),
                    value=heatmap_cfg["grow"],
                )

            # Decay
            heatmap.decay(heatmap_cfg["decay"])

            # Apply
            frame, heat_layer = heatmap.apply(
                image=frame,
                blurriness=heatmap_cfg["blur"],
                alpha=heatmap_cfg["alpha"],
            )

        # Human count
        if human_count_cfg:
            # Update new value
            human_counter.update(value=len(outputs))
            # Add to frame
            VIDEO.add_text(
                text=f"Person: {human_counter.get_value()}", pos=(20, 40), thickness=2
            )

        # Human loop
        for output in outputs:
            x1, y1, x2, y2 = output["box"]

            # Human box
            if cfg["detector"]["show"]["human_box"]:
                VIDEO.add_box(
                    top_left=(x1, y1), bottom_right=(x2, y2), color=255, thickness=2
                )

            # Human dot
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if cfg["detector"]["show"]["human_dot"]:
                cv2.circle(frame, center, 5, (225, 225, 225), -1)

            TrackBox.check(pos=center)

            # Classify action
            if cfg["classifier"]:
                X = TRANSFORM(output["human"])

                result = CLASSIFIER(X)

                VIDEO.add_text(text=result, pos=(x1, y1 - 5), thickness=2)

            # Tracker
            if cfg["detector"]["model"]["track"]:
                VIDEO.add_text(text=output["id"], pos=(x1, y1 - 5), thickness=2)

            # Confidence score
            if cfg["detector"]["show"]["score"]:
                VIDEO.add_text(
                    text=f"{output['conf']:.2}", pos=(x1, y2 - 5), thickness=2
                )

        # Show track box
        TrackBox.show(frame)

        # Show main video
        VIDEO.show()

        # Write output
        if writer_cfg:
            writer.write(frame)

        # Update progress
        progress_bar.update(1)

        # Delay and check keyboard input
        if not VIDEO.delay(cfg["video"]["delay"]):
            break

    # Release
    if heatmap_cfg:
        heatmap.release()
    if writer_cfg:
        writer.release()
    VIDEO.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
