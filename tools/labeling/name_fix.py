from tqdm import tqdm
import argparse
import os


def main(args):
    PATH: str = args.path

    files = list(
        name for name in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, name))
    )

    if not files:
        raise FileNotFoundError("Cannot found any file to fix name.")

    for file in tqdm(files):
        src_path = os.path.join(PATH, file)

        parent = PATH.split("/")[-1]

        dst_path = os.path.join(PATH, f"{parent}_{file}")

        os.rename(src_path, dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)
    args = parser.parse_args()
    main(args)
