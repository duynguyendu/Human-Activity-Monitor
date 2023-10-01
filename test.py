from argparse import ArgumentParser
from collections import Counter
import random
import os

from modules.transform import DataAugmentation
from modules.processing import VideoProcessing
from modules.model import LitModel
from models import ViT_B_16, VGG11

from lightning.pytorch import seed_everything
import torch

import cv2
import numpy as np
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
    sampling_value = 4, 
    num_frames = 0, 
    size = (750, 750)
)

DATA_PATH = "data/UTD-MHAD"

MODEL = VGG11(num_classes=27, hidden_features=512)

CHECKPOINT = "lightning_logs/VGG11_512_UTD/checkpoints/last.ckpt"



def main(args):
    # Define dataset
    extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.mpg']
    video_paths = []

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                video_path = os.path.join(root, file)
                video_paths.append(video_path)
    
    # Define classes
    classes = sorted(os.listdir("data/UTD-MHAD_x/test"))

    # Define model
    lit_model = LitModel(MODEL, checkpoint=CHECKPOINT).to(DEVICE)

    # Iterate random video
    for i, path in enumerate(random.sample(video_paths, 1)):
        results = []
        for frame in VP(path):
            with torch.inference_mode():
                X = TRANSFORM(Image.fromarray(frame)).unsqueeze(0).to(DEVICE)
                outputs = lit_model(X)
                _, pred = torch.max(outputs, 1)

            results.append(classes[pred.item()])

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # cv2.putText(
            #     frame, 
            #     classes[unique_items[most_frequent_index]], 
            #     (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
            #     2, (255, 0, 0), 5
            # )

            cv2.imshow("a", frame)

            if cv2.waitKey(100) == ord('q'):
                exit()

        element_counts = Counter(results)

        sorted_list = sorted(results, key=lambda x: (-element_counts[x], x), reverse=True)

        print(Counter(sorted_list))



if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--image", type=str, default=None)
    args = parser.parse_args()

    main(args)
