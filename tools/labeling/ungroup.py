from tqdm import tqdm
import argparse
import shutil
import os


def main(args):
    PATH: str = args.path

    folders = set(
        os.path.join(PATH, name)
        for name in os.listdir(PATH)
        if os.path.isdir(os.path.join(PATH, name))
    )

    if not folders:
        raise FileNotFoundError("Cannot found any folder to ungroup.")

    for idx in tqdm(folders):
        [
            shutil.move(os.path.join(idx, name), os.path.join(PATH, name))
            for name in os.listdir(idx)
        ]
        os.rmdir(idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)
    args = parser.parse_args()
    main(args)
