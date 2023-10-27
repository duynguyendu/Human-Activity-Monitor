import os, shutil
from rich.progress import track
from rootutils import autosetup
autosetup()



# Data classes
LABELS = {
    f"a{i}": action
    for i, action in enumerate(
        [
            "Swipe left", "Swipe right", "Wave", "Clap", "Throw",
            "Arm cross", "Basketball shoot", "Draw X", "Draw circle (forward)",
            "Draw circle (backward)", "Draw triangle", "Bowling", "Boxing",
            "Baseball swing", "Tennis swing", "Arm curl", "Tennis serve", "Push",
            "Knock", "Catch", "Pickup and throw", "Jog", "Walk", "Sit to stand",
            "Stand to sit", "Lunge", "Squat"
        ],
        start=1
    )
}

# UTD data path
DATA_PATH = "data/UTD-MAHD"



def main():
    data = os.listdir(DATA_PATH)

    for video in track(data):
        key = video.split("_")[0]
        dst_path = os.path.join(DATA_PATH, LABELS[key])
        os.makedirs(dst_path, exist_ok=True)
        shutil.move(
            os.path.join(DATA_PATH, video),
            os.path.join(dst_path, video)
        )



if __name__=="__main__":
    main()
