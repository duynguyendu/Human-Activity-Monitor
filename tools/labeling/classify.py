import argparse
import random
import shutil
import os

from skimage.metrics import structural_similarity
from tqdm import tqdm
import cv2


def ssim(image_1, image_2):
    gray_image1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    return structural_similarity(gray_image1, gray_image2)


def main(args):
    PATH: str = args.path

    THRESHOLD = args.threshold

    MIN_FILE = 5

    count = 0
    prev_run = []
    while True:
        files = list(
            name
            for name in os.listdir(PATH)
            if os.path.isfile(os.path.join(PATH, name))
        )

        # Stop if 0 file or total file not change
        if len(files) == 0 or (len(prev_run) == len(files)):
            break

        storage = set()

        # Prepare a key image to compare
        key_name = random.choice(files)
        key = cv2.imread(os.path.join(PATH, key_name))
        storage.add(key_name)

        # Check destiny folder
        while True:
            dst_folder = os.path.join(PATH, str(count))
            if not os.path.exists(dst_folder):
                break
            count += 1

        # Comparing loop
        for file in tqdm(files, desc=str(count)):
            current = cv2.imread(os.path.join(PATH, file))

            score = ssim(key, current)

            if score < THRESHOLD:
                continue

            storage.add(file)

        # Only move file if total > 5
        if len(storage) > MIN_FILE:
            os.makedirs(dst_folder)
            [
                shutil.move(os.path.join(PATH, file), os.path.join(dst_folder, file))
                for file in storage
            ]

        count + 1
        prev_run = files


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True)
    ap.add_argument("-th", "--threshold", type=float, required=True)
    args = ap.parse_args()
    main(args)
