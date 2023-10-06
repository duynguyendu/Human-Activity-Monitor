from argparse import ArgumentParser
from collections import Counter
import random
import os

from modules.transform import DataAugmentation
from modules.processing import VideoProcessing
from modules.model import LitModel
from modules.data import *
from models import *

from lightning.pytorch import seed_everything
import torch

import cv2
from PIL import Image
from rich import traceback, print
traceback.install()



# Set seed
# seed_everything(seed=42, workers=True)

# Set number of worker (CPU will be used | Default: 80%)
NUM_WOKER = int(os.cpu_count()*0.8) if torch.cuda.is_available() else 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRANSFORM = DataAugmentation().DEFAULT
VP = VideoProcessing(
    sampling_value = 2,
    max_frames = 0,
    min_frames = 0,
    size = (750, 750)
)


DATA_PATH = "data/UCF-101"
CLASESS = sorted(os.listdir(DATA_PATH))

SHOW_VIDEO = True
NUM_VIDEO = 3   # number of random video
WAITKEY = 45    # millisecond before next frame

MODEL = ViT_B_32(num_classes=len(CLASESS))

CHECKPOINT = "lightning_logs/ViT_B_32_UCF101/checkpoints/epoch=14-step=23415.ckpt"



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
                X = TRANSFORM(Image.fromarray(frame)).unsqueeze(0).to(DEVICE)
                outputs = lit_model(X)
                _, pred = torch.max(outputs, 1)

            result = CLASESS[pred.item()]

            total.append(result)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.putText(
                frame, result, 
                (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                2, (255, 0, 0), 5
            )

            cv2.imshow(path, frame) if SHOW_VIDEO else None

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
