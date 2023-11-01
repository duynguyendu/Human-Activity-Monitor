from skimage.metrics import structural_similarity
from tqdm import tqdm
import numpy as np
import cv2

# Setup root
import os, sys

sys.path.extend([os.getcwd(), f"{os.getcwd()}/src"])

from src.components.detectors.yolo_v8 import YoloV8


def ssim(image_1, image_2):
    gray_image1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    return structural_similarity(gray_image1, gray_image2)


def main():
    PATH = "/media/ht0710/Run/Data/Custom"

    THRESHOLD = 0.3

    SAMPLING = 5

    SAVE_PATH = f"data/processed/{THRESHOLD}"

    MODEL = YoloV8(conf=0.25, iou=0.3)

    data = os.listdir(PATH)

    for k, name in enumerate(data, 1):
        video_path = os.path.join(PATH, name)
        save_path = os.path.join(SAVE_PATH, name[:-4])

        if os.path.exists(save_path):
            continue

        os.makedirs(save_path, exist_ok=True)

        video = cv2.VideoCapture(video_path)

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        progress_bar = tqdm(
            total=total_frames,
            desc=f"{name} | {k}/{len(data)}",
            miniters=1,
            smoothing=0.1,
        )

        prev_run = []
        for i, (_, frame) in enumerate(iter(video.read, (False, None))):
            if i % SAMPLING == 0:
                outputs = MODEL(frame)

                for j, output in enumerate(outputs, 1):
                    if prev_run:
                        scores = set(
                            ssim(prev["human"], output["human"]) for prev in prev_run
                        )
                        best = max(scores)

                        prev_run.pop(np.argmax(best))

                        if best > THRESHOLD:
                            continue

                    cv2.imwrite(f"{save_path}/{i}_{j}.jpg", output["human"])

                prev_run = outputs

            progress_bar.update(1)


if __name__ == "__main__":
    main()
