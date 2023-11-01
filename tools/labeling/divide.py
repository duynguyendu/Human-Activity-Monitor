from tqdm import tqdm
import argparse
import shutil
import os


def main(args):
    PATH: str = args.path

    BREAKPOINT: int = args.breakpoint

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

    chunks = list(files[i : i + BREAKPOINT] for i in range(0, len(files), BREAKPOINT))

    count = 0
    for chunk in tqdm(chunks):
        dst_folder_path = os.path.join(PATH, str(count))

        os.makedirs(dst_folder_path, exist_ok=True)

        for file in chunk:
            shutil.move(os.path.join(PATH, file), os.path.join(dst_folder_path, file))

            count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("-bp", "--breakpoint", type=int, required=True)
    args = parser.parse_args()
    main(args)
