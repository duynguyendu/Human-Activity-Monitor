from argparse import ArgumentParser
from collections import Counter
import random
import os

from modules.data import VideoProcessing
from modules.model import LitModel
from models import ViT_B_16

from lightning.pytorch import seed_everything
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch

import cv2
import numpy as np
from rich import traceback, print
traceback.install()


# Set seed
# seed_everything(seed=42, workers=True)

# Set number of worker (CPU will be used | Default: 80%)
NUM_WOKER = int(os.cpu_count()*0.8) if torch.cuda.is_available() else 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224), antialias=True),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])



def main(args):
    DATA_PATH = "data/UCF11"

    extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.mpg']

    video_paths = []

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                video_path = os.path.join(root, file)
                video_paths.append(video_path)

    classes = sorted(os.listdir("data/UCF11"))

    # Define model
    model = ViT_B_16(num_classes=11)
    lit_model = LitModel(
        model = model,
        checkpoint = "lightning_logs/version_3/checkpoints/epoch=9-step=8370.ckpt"
    ).to(DEVICE)

    VP = VideoProcessing(1, 0, (750, 750))

    for path in random.sample(video_paths, 3):
        results = []
        for frame in VP(path):
            with torch.inference_mode():
                X = transform(frame).unsqueeze(0).to(DEVICE)
                outputs = lit_model(X)
                _, pred = torch.max(outputs, 1)

            results.append(pred.item())
            if len(results) > 4:
                results = sorted(results, key=lambda x: Counter(results)[x])
                results = results[1:-1]

            unique_items, counts = np.unique(results, return_counts=True)
            most_frequent_index = np.argmax(counts)

            cv2.putText(
                frame, 
                classes[unique_items[most_frequent_index]], 
                (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                2, (255, 0, 0), 5
            )

            cv2.imshow("a", frame)
            if cv2.waitKey(1) == ord('q'):
                break



if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--image", type=str, default=None)
    args = parser.parse_args()

    main(args)
