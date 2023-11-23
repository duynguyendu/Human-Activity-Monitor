from skimage.metrics import structural_similarity
from tqdm import tqdm
import numpy as np
import cv2
import os

# Setup root
from rootutils import autosetup

autosetup()

from src.modules.data import ImageProcessing
from src.modules.utils import tuple_handler
from src.components import Detector


def ssim(image_1, image_2):
    gray_image1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    return structural_similarity(gray_image1, gray_image2)


def main():
    PATH = "/media/ht0710/Data/Data/Custom"

    THRESHOLD = 0.3

    SAMPLING = 5

    MARGIN = 30

    SIZE = 224

    SAVE_PATH = f"data/processed/{THRESHOLD}"

    MODEL = Detector(weight="weights/yolov8x.pt", half=True, fuse=True, optimize=True)

    data = sorted(
        os.listdir(PATH),
        key=lambda x: tuple(int(i) for i in x.split(".")[0].split("_")),
    )

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

                current = []
                for j, output in enumerate(outputs, 1):
                    x1, y1, x2, y2 = output["box"]

                    x1 = max(0, x1 - MARGIN)
                    y1 = max(0, y1 - MARGIN)
                    x2 = min(frame.shape[1], x2 + MARGIN)
                    y2 = min(frame.shape[1], y2 + MARGIN)

                    human = frame[y1:y2, x1:x2]

                    human = ImageProcessing.add_border(human)

                    human = cv2.resize(human, tuple_handler(SIZE))

                    if prev_run:
                        scores = set(ssim(prev, human) for prev in prev_run)
                        best = max(scores)

                        prev_run.pop(np.argmax(best))

                        if best > THRESHOLD:
                            continue

                    current.append(human)

                    cv2.imwrite(f"{save_path}/{i}_{j}.jpg", human)

                prev_run = current

            progress_bar.update(1)


if __name__ == "__main__":
    main()
