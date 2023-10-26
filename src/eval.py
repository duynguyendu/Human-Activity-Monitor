from argparse import ArgumentParser
from collections import Counter
import random
import os

from modules.transform import DataTransformation
from modules.processing import VideoProcessing
from modules.model import LitModel
from modules.data import *
from models import *

from ultralytics import YOLO
from lightning.pytorch import seed_everything
import torch

import cv2
from PIL import Image
from rich import traceback, print
traceback.install()



# Set seed
# seed_everything(seed=42, workers=True)

# Set number of worker (CPU will be used | Default: 80%)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRANSFORM = DataTransformation().TOPIL
VP = VideoProcessing(
    sampling_value = 2,
    max_frame = 0,
    min_frame = 0,
    add_border = False,
    size = (750, 750)
)


DATA_PATH = "data/UTD-MHAD"
CLASESS = sorted(os.listdir("data/UCF101/test"))

SHOW_VIDEO = True
NUM_VIDEO = 10   # number of random video
WAITKEY = 100    # millisecond before next frame

EXTRACTOR = YOLO("yolov8x")

MODEL = ViT(version="B_32", num_classes=101)

CHECKPOINT = "logs/new/ViT_B_32_ft_UCF101/checkpoints/epoch=5-step=8544.ckpt"



def main(args):
    # Define dataset
    extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.mpg']
    video_paths = []

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                video_path = os.path.join(root, file)
                video_paths.append(video_path)

    # Define model
    lit_model = LitModel(MODEL, checkpoint=CHECKPOINT).to(DEVICE)

    # Iterate random video
    for i, path in enumerate(random.sample(video_paths, NUM_VIDEO)):
        total = []
        for frame in VP(path):
            with torch.inference_mode():
                out = EXTRACTOR(frame, conf=0.25, iou=0.3, classes=0, verbose=False)[0]

                x1, y1, x2, y2, conf, _ = (int(i.item()) for i in out.boxes[0].data[0])

                human = frame[y1:y2, x1:x2]

                X = TRANSFORM(human).unsqueeze(0).to(DEVICE)

                outputs = lit_model(X)
                _, pred = torch.max(outputs, 1)

            result = CLASESS[pred.item()]

            total.append(result)

            if SHOW_VIDEO:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                cv2.putText(
                    frame, result, 
                    (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (255, 0, 0), 5
                )

                cv2.imshow(path, frame)

                key = cv2.waitKey(WAITKEY) & 0xFF
                if key == ord('c'):
                    break
                if key == ord('q'):
                    exit()

        sorted_list = sorted(total, key=lambda x: (-Counter(total)[x], x), reverse=True)

        print("\n[bold][green]Path:[/][/]", f"[white]{path}[/]")
        print(Counter(sorted_list))

        cv2.destroyAllWindows()



if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--image", type=str, default=None)
    args = parser.parse_args()

    main(args)
