from argparse import ArgumentParser
import numpy as np
from rich import traceback
traceback.install()
import os

import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from lightning.pytorch import seed_everything
from modules.data import VideoProcessing
from models.VGG import VGG19
import cv2


# Set seed
# seed_everything(seed=42, workers=True)

# Set number of worker (CPU will be used | Default: 80%)
NUM_WOKER = int(os.cpu_count()*0.8) if torch.cuda.is_available() else 0
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
    # Define dataset
    dataset = ImageFolder(root="data/UCF11_x")
    classes = dataset.classes

    # Define model
    model = VGG19(num_classes=11, hidden_features=256)
    model.load("HAR/9e2h6wnn/checkpoints/last.ckpt")

    VP = VideoProcessing(2, 0, (700, 700))

    video = VP("data/UCF11/swing/v_swing_05/v_swing_05_04.mpg")


    results = []
    for frame in video:

        X = transform(frame)
        with torch.inference_mode():
            outputs = model(X.unsqueeze(0))
            _, pred = torch.max(outputs, 1)
            results.append(pred.item())
        if len(results) >= 10:
            results.pop()

        cv2.putText(
            frame, 
            classes[int(np.mean(results))], 
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (255, 0, 0), 3
        )


        cv2.imshow("a", frame)
        if cv2.waitKey(1) == ord('q'):
            break




if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--image", type=str, default=None)
    args = parser.parse_args()

    main(args)
