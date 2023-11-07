from tqdm import tqdm
import argparse
import shutil
import os


def main(args):
    PATH: str = args.path

    files = sorted(
        list(
            name
            for name in os.listdir(PATH)
            if os.path.isfile(os.path.join(PATH, name))
        ),
        key=lambda x: int(x.split("_")[0]),
    )

    if not files:
        raise FileNotFoundError("Cannot found any file to group.")

    for file in tqdm(files):
        src_path = os.path.join(PATH, file)

        dest_folder_path = os.path.join(PATH, file[:-4].split("_")[-1])

        os.makedirs(dest_folder_path, exist_ok=True)

        shutil.move(src_path, os.path.join(dest_folder_path, file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)
    args = parser.parse_args()
    main(args)
