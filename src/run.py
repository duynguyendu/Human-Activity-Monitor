import os
import shutil

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


@hydra.main(config_path="../configs", config_name="run", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Remove the hydra outputs since we already have lightning logs
    shutil.rmtree("outputs")

    # Define main componets
    # Detector
    detector = YoloV8(
        **cfg["detector"], size=cfg["video"]["image_size"], device=cfg["device"]
    )

    # Transform
    transfrom = DataTransformation.TOPIL(image_size=cfg["video"]["image_size"])

    # Classifier
    classifier = ViT(checkpoint=cfg["classifier"]["checkpoint"], device=cfg["device"])

    # Open video
    video = Video(path=cfg["video"]["path"])

    # Config video progress bar
    progress_bar = tqdm(
        total=video.total_frame,
        desc=f"  {video.name}",
        unit=" frame",
        miniters=1,
        smoothing=0.1,
        delay=0.5,
    )

    # Frame loop
    for frame in video.get_frame():
        # Frame sampling
        if progress_bar.n % max(1, cfg["video"]["sampling"]) != 0:
            progress_bar.update(1)
            continue

        # First run setup
        if progress_bar.n == 0:
            print("[bold]Initialize:[/]")
            add_text = lambda text, pos: cv2.putText(
                frame,
                text,
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Initialize video writer
            writer_cfg = cfg["video"]["save_output"]
            if writer_cfg["enable"]:
                save_path = os.path.join(
                    writer_cfg["save_path"],
                    video.stem,
                    writer_cfg["save_name"] + ".mp4",
                )
                writer = video.writer(save_path=save_path)
                print(f"  [bold]Saving output to:[/] [green]{save_path}[/]")

            # Initialize person count
            human_count_cfg = cfg["feature"]["person_count"]
            if human_count_cfg["enable"]:
                human_counter = HumanCount(smoothness=human_count_cfg["smoothness"])
                human_counter.save_config(
                    save_path=os.path.join(
                        human_count_cfg["save"]["save_path"],
                        video.stem,
                        human_count_cfg["save"]["save_name"] + ".csv",
                    ),
                    interval=human_count_cfg["save"]["interval"],
                    sampling=cfg["video"]["sampling"],
                )

            # Initialize heatmap
            heatmap_cfg = cfg["feature"]["heatmap"]
            if heatmap_cfg["enable"]:
                heatmap = Heatmap(shape=video.size(reverse=True))

                # Heatmap config save
                hm_save_path = os.path.join(
                    heatmap_cfg["save"]["save_path"],
                    video.stem,
                    heatmap_cfg["save"]["save_name"],
                )
                if heatmap_cfg["save"]["video"]:
                    heatmap.save_video(
                        save_path=hm_save_path + ".mp4",
                        fps=video.fps,
                        size=video.size(),
                    )
                    print(
                        f"  [bold]Saving heatmap video to:[/] [green]{hm_save_path}.mp4[/]"
                    )
                if heatmap_cfg["save"]["image"]:
                    heatmap.save_image(
                        save_path=hm_save_path + ".jpg",
                        size=video.size(reverse=True),
                    )
                    print(
                        f"  [bold]Saving heatmap image to:[/] [green]{hm_save_path}.jpg[/]"
                    )
            print(f"[bold]Video progress:[/]")

        # Detect human
        outputs = detector(frame)

        # Heatmap
        if heatmap_cfg["enable"]:
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
        if human_count_cfg["enable"]:
            # Update new value
            human_counter.update(value=len(outputs))
            # Add to frame
            add_text(f"Person: {human_counter.get_value()}", pos=(20, 40))

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
            if cfg["classifier"]["enable"]:
                X = transfrom(output["human"])

                result = classifier(X)

                add_text(result, pos=(x1, y1 - 5))

        cv2.imshow(video.stem, frame)

        if writer_cfg["enable"]:
            writer.write(frame)

        progress_bar.update(1)

        if not video.delay(cfg["video"]["delay"]):
            break

    (x.release() for x in (video, writer, heatmap))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
